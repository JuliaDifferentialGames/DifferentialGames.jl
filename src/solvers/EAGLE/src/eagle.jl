"""
    GNEPSolution{T}

General solution structure for all GNEP types (LQ games, potential games, etc.).

# Fields
- `strategies::Any` : Strategy representation (problem-dependent)
- `costs::Vector{T}` : Final costs for each player
- `metadata::Dict{Symbol, Any}` : Additional solver-specific information
"""
struct GNEPSolution{T}
    strategies::Any              # Flexible strategy storage
    costs::Vector{T}             # Cost for each player
    metadata::Dict{Symbol, Any}  # Solver-specific data
end

# Convenience constructor
function GNEPSolution(strategies, costs::Vector{T}; metadata...) where T
    GNEPSolution{T}(strategies, costs, Dict{Symbol, Any}(metadata...))
end

"""
    FeedbackStrategy{T}

Affine feedback strategy: u(t) = P(t)*x(t) + α(t)

# Fields
- `P::Vector{Matrix{T}}` : feedback gain matrices P(k) for each timestep
- `α::Vector{Vector{T}}` : feedforward terms α(k) for each timestep
- `dt::T` : discretization timestep
- `tf::T` : final time horizon
"""
struct FeedbackStrategy{T}
    P::Vector{Matrix{T}}      # Joint feedback gains [P(1), P(2), ..., P(T)]
    α::Vector{Vector{T}}      # Feedforward terms [α(1), α(2), ..., α(T)]
    dt::T                      # Time discretization
    tf::T                      # Time horizon
end

# Convenience accessor methods
Base.length(fs::FeedbackStrategy) = length(fs.P)
timesteps(fs::FeedbackStrategy) = length(fs.P)

"""
    evaluate(strategy::FeedbackStrategy, x, k)

Evaluate feedback strategy at timestep k: u = P(k)*x + α(k)
"""
function evaluate(strategy::FeedbackStrategy, x::AbstractVector, k::Int)
    return strategy.P[k] * x + strategy.α[k]
end

"""
    evaluate(strategy::FeedbackStrategy, x, t)

Evaluate feedback strategy at continuous time t (interpolates to nearest timestep).
"""
function evaluate(strategy::FeedbackStrategy, x::AbstractVector, t::Real)
    k = clamp(round(Int, t / strategy.dt) + 1, 1, length(strategy))
    return evaluate(strategy, x, k)
end

"""
    EAGLE <: AbstractLQSolver

**E**fficient **A**lgorithm for **G**ame-theoretic **L**Q **E**quilibria

A dynamic programming solver for finite-horizon linear-quadratic differential games.
Computes feedback Nash equilibrium strategies using backward Riccati recursion with:
- Schur complement method for exploiting block-diagonal structure
- Joseph-form updates for numerical stability
- Efficient factorization to minimize allocations

# Fields
- `dt::Float64` : discretization timestep
- `max_schur_iters::Int` : maximum iterations for Schur complement method (default: 3)
- `schur_tol::Float64` : convergence tolerance for Schur iterations (default: 1e-8)
- `verbose::Bool` : print convergence information (default: false)

# Example
```julia
game = LQGameProblem(A, B, Q, R, Qf, x0, tf, control_dims)
solver = EAGLE(0.1)  # 0.1 second timestep
solution = solve(game, solver)

# Extract strategies
strategy = solution.strategies
costs = solution.costs

# Evaluate control at timestep k
u_k = evaluate(strategy, x_k, k)
```
"""
struct EAGLE <: AbstractLQSolver
    dt::Float64
    max_schur_iters::Int
    schur_tol::Float64
    verbose::Bool
end

# Convenience constructors
EAGLE(dt::Float64) = EAGLE(dt, 3, 1e-8, false)
EAGLE(dt::Float64, verbose::Bool) = EAGLE(dt, 3, 1e-8, verbose)

"""
    EAGLECache

Pre-allocated workspace for EAGLE solver to minimize allocations.
"""
struct EAGLECache{T, N, M, NP}
    # Cost-to-go representation (Zi = Li*Li', not stored explicitly)
    L::Vector{Matrix{T}}       # Cholesky factors, length NP, each N×N
    ζ::Vector{Vector{T}}       # Linear cost-to-go, length NP, each N
    
    # Working matrices for S*P = Y system
    S::Matrix{T}               # M×M
    YP::Matrix{T}              # M×N
    Yα::Vector{T}              # M
    
    # Block diagonal components
    S_blocks::Vector{Matrix{T}}  # Diagonal blocks Sᵢᵢ
    
    # Intermediate results
    F::Matrix{T}               # N×N, closed-loop dynamics
    β::Vector{T}               # N
    
    # Temporaries for updates
    temp_mat::Matrix{T}        # N×N
    temp_vec::Vector{T}        # N
    BᵢZᵢ::Matrix{T}            # For Bᵢ'*Zᵢ computation
    
    # Previous iteration for Schur method
    P_prev::Matrix{T}          # M×N
    α_prev::Vector{T}          # M
end

function EAGLECache(::Type{T}, n::Int, m::Int, np::Int, control_dims::Vector{Int}) where T
    EAGLECache{T, n, m, np}(
        [zeros(T, n, n) for _ in 1:np],
        [zeros(T, n) for _ in 1:np],
        zeros(T, m, m),
        zeros(T, m, n),
        zeros(T, m),
        [zeros(T, control_dims[i], control_dims[i]) for i in 1:np],
        zeros(T, n, n),
        zeros(T, n),
        zeros(T, n, n),
        zeros(T, n),
        zeros(T, m, n),
        zeros(T, m, n),
        zeros(T, m)
    )
end

"""
    solve(prob::LQGameProblem, solver::EAGLE)

Solve the finite-horizon LQ game using EAGLE.
Returns a `GNEPSolution` with feedback strategies.

# Arguments
- `prob::LQGameProblem` : the LQ game problem
- `solver::EAGLE` : the EAGLE solver instance

# Returns
- `GNEPSolution` containing:
  - `strategies::FeedbackStrategy` : time-varying feedback gains
  - `costs::Vector` : cost-to-go at initial state for each player
  - `metadata` : solver information (iterations, convergence, etc.)
"""
function solve(prob::LQGameProblem{T, N, M, NP}, solver::EAGLE) where {T, N, M, NP}
    # Discretize time horizon
    num_steps = ceil(Int, prob.tf / solver.dt)
    dt = prob.tf / num_steps
    
    solver.verbose && println("EAGLE solver: discretizing with $num_steps timesteps (dt=$(dt))")
    
    # Discretize dynamics: xₖ₊₁ = Ad*xₖ + Bd*uₖ
    Ad, Bd = discretize_dynamics(prob.A, prob.B, dt)
    
    # Discretize costs (scale by dt for running costs)
    Qd = [dt * Q for Q in prob.Q]
    Rd = [dt * R for R in prob.R]
    
    # Initialize cache
    cache = EAGLECache(T, N, M, NP, prob.control_dims)
    
    # Allocate solution storage
    P_traj = [zeros(T, M, N) for _ in 1:num_steps]
    α_traj = [zeros(T, M) for _ in 1:num_steps]
    
    # Run backward DP
    solver.verbose && println("EAGLE solver: running backward pass...")
    convergence_history = solve_backward_pass!(
        P_traj, α_traj, cache, Ad, Bd, Qd, Rd, prob.Qf, 
        prob.control_dims, num_steps, solver
    )
    
    # Compute costs at initial state
    costs = compute_initial_costs(cache.L, cache.ζ, prob.x0, NP)
    
    solver.verbose && println("EAGLE solver: complete. Player costs: ", costs)
    
    # Build feedback strategy
    strategy = FeedbackStrategy{T}(P_traj, α_traj, dt, prob.tf)
    
    # Create solution with metadata
    return GNEPSolution(
        strategy, 
        costs,
        solver_name = :EAGLE,
        num_timesteps = num_steps,
        dt = dt,
        convergence_history = convergence_history,
        num_players = NP,
        state_dim = N,
        control_dim = M
    )
end

"""
    discretize_dynamics(A, B, dt)

Discretize continuous-time dynamics using matrix exponential.
"""
function discretize_dynamics(A::Matrix{T}, B::Vector{Matrix{T}}, dt::T) where T
    n = size(A, 1)
    m_total = sum(size(Bi, 2) for Bi in B)
    
    # Form augmented matrix [A B; 0 0]
    B_full = hcat(B...)
    aug = [A B_full; zeros(T, m_total, n + m_total)]
    
    # Matrix exponential
    exp_aug = exp(dt * aug)
    
    Ad = exp_aug[1:n, 1:n]
    Bd_full = exp_aug[1:n, n+1:end]
    
    # Split Bd back into per-player matrices
    Bd = Vector{Matrix{T}}(undef, length(B))
    col_idx = 1
    for i in 1:length(B)
        mi = size(B[i], 2)
        Bd[i] = Bd_full[:, col_idx:col_idx+mi-1]
        col_idx += mi
    end
    
    return Ad, Bd
end

"""
    solve_backward_pass!

Main backward dynamic programming loop.
Returns convergence history for Schur iterations.
"""
function solve_backward_pass!(P_traj, α_traj, cache, A, B, Q, R, Qf, control_dims, 
                              num_steps, solver)
    @unpack L, ζ, S, YP, Yα, S_blocks, F, β, temp_mat, temp_vec, BᵢZᵢ, P_prev, α_prev = cache
    
    NP = length(Q)
    N = size(A, 1)
    M = sum(control_dims)
    
    # Initialize terminal cost-to-go: Zᵢ(T) = Qfᵢ, ζᵢ(T) = 0
    for ii in 1:NP
        L[ii] .= cholesky(Hermitian(Qf[ii])).L
        fill!(ζ[ii], 0)
    end
    
    # Track convergence history
    convergence_history = Vector{Int}(undef, num_steps)
    
    # Backward pass
    for k in num_steps:-1:1
        # Build S and Y matrices
        build_S_and_Y!(S, YP, Yα, S_blocks, L, ζ, A, B, Q, R, control_dims, 
                       temp_mat, BᵢZᵢ, NP)
        
        # Solve for feedback gains using Schur complement method
        iters = solve_schur_complement!(P_traj[k], α_traj[k], S, YP, Yα, S_blocks, 
                                        control_dims, P_prev, α_prev, solver)
        convergence_history[k] = iters
        
        # Compute closed-loop dynamics F = A - B*P
        compute_closed_loop!(F, β, A, B, P_traj[k], α_traj[k], M, N)
        
        # Update cost-to-go using Joseph form
        update_cost_to_go_joseph!(L, ζ, F, β, P_traj[k], α_traj[k], Q, R, 
                                   control_dims, temp_mat, temp_vec, NP)
    end
    
    return convergence_history
end

"""
    build_S_and_Y!

Construct the linear system S*P = Y for determining feedback gains.
Exploits block diagonal structure of S.
"""
function build_S_and_Y!(S, YP, Yα, S_blocks, L, ζ, A, B, Q, R, control_dims, 
                        temp_mat, BᵢZᵢ, NP)
    fill!(S, 0)
    fill!(YP, 0)
    fill!(Yα, 0)
    
    B_full = hcat(B...)
    N = size(A, 1)
    
    # Build system for each player
    u_start = 1
    for ii in 1:NP
        mi = control_dims[ii]
        u_end = u_start + mi - 1
        udxᵢ = u_start:u_end
        
        # Reconstruct Zᵢ = Lᵢ*Lᵢ' for this player
        mul!(temp_mat, L[ii], L[ii]')
        
        # Compute Bᵢ'*Zᵢ
        mul!(BᵢZᵢ, temp_mat, B[ii])  # Zᵢ*Bᵢ
        BᵢZᵢ_T = BᵢZᵢ'  # Bᵢ'*Zᵢ (as transpose)
        
        # Diagonal block: Sᵢᵢ = Rᵢ + Bᵢ'*Zᵢ*Bᵢ
        mul!(S_blocks[ii], BᵢZᵢ_T, B[ii])
        S_blocks[ii] .+= R[ii]
        S[udxᵢ, udxᵢ] .= S_blocks[ii]
        
        # Off-diagonal blocks: Sᵢⱼ = Bᵢ'*Zᵢ*Bⱼ
        mul!(S[udxᵢ, :], BᵢZᵢ_T, B_full)
        
        # Right-hand side for P: YPᵢ = Bᵢ'*Zᵢ*A
        mul!(YP[udxᵢ, :], BᵢZᵢ_T, A)
        
        # Right-hand side for α: Yαᵢ = Bᵢ'*ζᵢ
        mul!(Yα[udxᵢ], B[ii]', ζ[ii])
        
        u_start = u_end + 1
    end
end

"""
    solve_schur_complement!

Solve S*P = YP and S*α = Yα using sequential Schur complement method.
This exploits block diagonal dominance and often converges in 1-2 iterations.

Returns the number of iterations used.
"""
function solve_schur_complement!(P, α, S, YP, Yα, S_blocks, control_dims, 
                                 P_prev, α_prev, solver)
    NP = length(S_blocks)
    M = size(S, 1)
    N = size(YP, 2)
    
    # Initialize with block-diagonal solution
    u_start = 1
    for ii in 1:NP
        mi = control_dims[ii]
        u_end = u_start + mi - 1
        udxᵢ = u_start:u_end
        
        # Initial guess: solve diagonal blocks
        P[udxᵢ, :] .= S_blocks[ii] \ YP[udxᵢ, :]
        α[udxᵢ] .= S_blocks[ii] \ Yα[udxᵢ]
        
        u_start = u_end + 1
    end
    
    # Refine with Schur complement iterations
    iters_used = 1
    for iter in 1:solver.max_schur_iters
        copyto!(P_prev, P)
        copyto!(α_prev, α)
        
        u_start = 1
        for ii in 1:NP
            mi = control_dims[ii]
            u_end = u_start + mi - 1
            udxᵢ = u_start:u_end
            udx_rest = [1:u_start-1; u_end+1:M]
            
            if isempty(udx_rest)
                u_start = u_end + 1
                continue
            end
            
            # Schur complement update: solve for player ii given others
            # Sᵢᵢ*Pᵢ = YPᵢ - Sᵢ,rest*P_rest
            rhs_P = YP[udxᵢ, :] - S[udxᵢ, udx_rest] * P[udx_rest, :]
            rhs_α = Yα[udxᵢ] - S[udxᵢ, udx_rest] * α[udx_rest]
            
            P[udxᵢ, :] .= S_blocks[ii] \ rhs_P
            α[udxᵢ] .= S_blocks[ii] \ rhs_α
            
            u_start = u_end + 1
        end
        
        # Check convergence
        if iter > 1
            ΔP = norm(P - P_prev)
            Δα = norm(α - α_prev)
            if max(ΔP, Δα) < solver.schur_tol
                iters_used = iter
                break
            end
        end
        iters_used = iter + 1
    end
    
    return iters_used
end

"""
    compute_closed_loop!

Compute closed-loop dynamics F = A - B*P and feedforward β = -B*α.
"""
function compute_closed_loop!(F, β, A, B, P, α, M, N)
    B_full = hcat(B...)
    
    # F = A - B*P
    mul!(F, B_full, P, -1.0, 0.0)
    F .+= A
    
    # β = -B*α
    mul!(β, B_full, α, -1.0, 0.0)
end

"""
    update_cost_to_go_joseph!

Update cost-to-go using Joseph form for numerical stability.
Maintains Cholesky factorization Zᵢ = Lᵢ*Lᵢ' throughout.
"""
function update_cost_to_go_joseph!(L, ζ, F, β, P, α, Q, R, control_dims, 
                                   temp_mat, temp_vec, NP)
    N = size(F, 1)
    
    u_start = 1
    for ii in 1:NP
        mi = control_dims[ii]
        u_end = u_start + mi - 1
        udxᵢ = u_start:u_end
        
        # Update Zᵢ = F'*Zᵢ*F + Qᵢ + Pᵢ'*Rᵢ*Pᵢ using Cholesky factors
        # Compute Z explicitly then factor
        mul!(temp_mat, L[ii], L[ii]')  # Zᵢ(k+1)
        
        # F'*Z*F
        Z_new = F' * temp_mat * F
        
        # Add Q
        Z_new .+= Q[ii]
        
        # Add P'*R*P
        Pᵢ = P[udxᵢ, :]
        Z_new .+= Pᵢ' * R[ii] * Pᵢ
        
        # Factor the result
        L[ii] .= cholesky(Hermitian(Z_new)).L
        
        # Update ζᵢ = F'*(ζᵢ + Zᵢ*β) + lᵢ + Pᵢ'*Rᵢ*αᵢ
        # Note: lᵢ = 0 for standard LQ, rᵢ = 0 for standard LQ
        mul!(temp_vec, temp_mat, β)  # Zᵢ*β
        temp_vec .+= ζ[ii]            # ζᵢ + Zᵢ*β
        mul!(ζ[ii], F', temp_vec)     # F'*(ζᵢ + Zᵢ*β)
        
        # Add Pᵢ'*Rᵢ*αᵢ term
        αᵢ = α[udxᵢ]
        mul!(temp_vec, R[ii], αᵢ)    # Rᵢ*αᵢ
        mul!(ζ[ii], Pᵢ', temp_vec, 1.0, 1.0)  # += Pᵢ'*Rᵢ*αᵢ
        
        u_start = u_end + 1
    end
end

"""
    compute_initial_costs

Compute J_i(x0) = x0'*Z_i(0)*x0 + x0'*ζ_i(0) for each player.
"""
function compute_initial_costs(L, ζ, x0, NP)
    costs = zeros(NP)
    temp = similar(x0)
    
    for ii in 1:NP
        # Reconstruct Z = L*L'
        # x0'*Z*x0 = x0'*L*L'*x0 = ||L'*x0||²
        mul!(temp, L[ii]', x0)
        costs[ii] = dot(temp, temp) + dot(x0, ζ[ii])
    end
    
    return costs
end

# ============================================================================
# Utility functions for working with solutions
# ============================================================================

"""
    simulate(prob::LQGameProblem, solution::GNEPSolution)

Simulate the closed-loop system using the computed Nash equilibrium strategies.

Returns trajectory of states and controls.
"""
function simulate(prob::LQGameProblem{T, N, M, NP}, solution::GNEPSolution) where {T, N, M, NP}
    strategy = solution.strategies
    num_steps = length(strategy)
    dt = strategy.dt
    
    # Discretize dynamics
    Ad, Bd = discretize_dynamics(prob.A, prob.B, dt)
    B_full = hcat(Bd...)
    
    # Allocate trajectory storage
    x_traj = [zeros(T, N) for _ in 1:num_steps+1]
    u_traj = [zeros(T, M) for _ in 1:num_steps]
    
    # Initial state
    x_traj[1] = prob.x0
    
    # Forward simulation
    for k in 1:num_steps
        # Compute control
        u_traj[k] = evaluate(strategy, x_traj[k], k)
        
        # Propagate dynamics
        x_traj[k+1] = Ad * x_traj[k] + B_full * u_traj[k]
    end
    
    return (states = x_traj, controls = u_traj, times = collect(0:dt:strategy.tf))
end

"""
    compute_trajectory_costs(prob::LQGameProblem, trajectory)

Compute the actual costs incurred by each player along a trajectory.
"""
function compute_trajectory_costs(prob::LQGameProblem{T, N, M, NP}, trajectory) where {T, N, M, NP}
    x_traj = trajectory.states
    u_traj = trajectory.controls
    dt = trajectory.times[2] - trajectory.times[1]
    
    costs = zeros(T, NP)
    
    u_start = 1
    for ii in 1:NP
        mi = prob.control_dims[ii]
        u_end = u_start + mi - 1
        
        # Running cost
        for k in 1:length(u_traj)
            x = x_traj[k]
            u = u_traj[k][u_start:u_end]
            costs[ii] += dt * (x' * prob.Q[ii] * x + u' * prob.R[ii] * u)
        end
        
        # Terminal cost
        xf = x_traj[end]
        costs[ii] += xf' * prob.Qf[ii] * xf
        
        u_start = u_end + 1
    end
    
    return costs
end