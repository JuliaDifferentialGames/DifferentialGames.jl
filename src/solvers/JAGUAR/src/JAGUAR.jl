"""
    JAGUAR{T, NP}

Main JAGUAR solver for PD-GNEP problems.
"""
mutable struct JAGUAR{T, NP}
    config::JAGUARConfig{T}
    problem::PDGNEProblem{T, NP}
    
    # Computed parameters
    K::Int                # Timesteps per segment
    K_exec::Int           # Timesteps to execute before replanning
    
    # State
    ξ_prev::Union{Nothing, JointTrajectory{T}}  # Previous solution for warm-starting
    current_time::T
    
    # Diagnostics
    diagnostics::JAGUARDiagnostics{T}
    
    function JAGUAR{T, NP}(
        config::JAGUARConfig{T},
        problem::PDGNEProblem{T, NP}
    ) where {T, NP}
        K = Int(ceil(config.T_segment / config.dt))
        K_exec = Int(ceil(config.T_execute / config.dt))
        
        @assert K_exec < K "Execution horizon must be less than planning horizon"
        
        new{T, NP}(
            config, problem, K, K_exec,
            nothing, zero(T),
            JAGUARDiagnostics{T}()
        )
    end
end

function JAGUAR(config::JAGUARConfig{T}, problem::PDGNEProblem{T, NP}) where {T, NP}
    return JAGUAR{T, NP}(config, problem)
end

# ============================================================================
# Core Algorithm
# ============================================================================

"""
    solve!(jaguar::JAGUAR{T, NP}) -> JointTrajectory{T}

Main solving loop: receding horizon planning with adaptive phase switching.
"""
function solve!(jaguar::JAGUAR{T, NP}) where {T, NP}
    config = jaguar.config
    prob = jaguar.problem
    
    # Initialize trajectories
    all_trajectories = JointTrajectory{T}[]
    
    config.verbose && println("=== JAGUAR Solver ===")
    config.verbose && println("Total horizon: $(prob.tf), Segment: $(config.T_segment)")
    
    segment_count = 0
    
    while jaguar.current_time < prob.tf
        segment_count += 1
        t_start = time()
        
        config.verbose && println("\n--- Segment $segment_count (t=$(jaguar.current_time)) ---")
        
        # Solve one planning segment
        ξ_segment = solve_segment!(jaguar)
        
        # Store diagnostics
        t_elapsed = time() - t_start
        push!(jaguar.diagnostics.computation_times, t_elapsed)
        push!(jaguar.diagnostics.costs_per_segment, ξ_segment.total_cost)
        
        config.verbose && println("Segment solved in $(round(t_elapsed*1000, digits=1))ms")
        config.verbose && println("Total cost: $(round(ξ_segment.total_cost, digits=3))")
        
        # Execute first portion of trajectory
        execute_and_advance!(jaguar, ξ_segment)
        
        push!(all_trajectories, ξ_segment)
    end
    
    config.verbose && println("\n=== JAGUAR Complete ===")
    config.verbose && println("Segments: $segment_count")
    config.verbose && println("Avg time: $(round(mean(jaguar.diagnostics.computation_times)*1000, digits=1))ms")
    
    # Concatenate all segments
    return concatenate_trajectories(all_trajectories)
end

"""
    solve_segment!(jaguar::JAGUAR{T, NP}) -> JointTrajectory{T}

Solve a single planning segment using adaptive phase switching.
"""
function solve_segment!(jaguar::JAGUAR{T, NP}) where {T, NP}
    config = jaguar.config
    prob = jaguar.problem
    
    # ========================================
    # INITIALIZATION: Learned/Warm-Start
    # ========================================
    
    ξ = initialize_trajectories(jaguar)
    
    # ========================================
    # ITERATIVE BEST RESPONSE
    # ========================================
    
    iter = 0
    residual = Inf
    use_ilqgames = false
    phase_switch_iter = -1
    
    while residual > config.ε_nash && iter < config.max_iter_coarse + config.max_iter_fine
        iter += 1
        
        # Decide which solver to use
        if residual < config.ε_transition || use_ilqgames
            use_ilqgames = true
            if phase_switch_iter < 0
                phase_switch_iter = iter
            end
        end
        
        if use_ilqgames
            # ========================================
            # PHASE 2: iLQGames (Fast Local Refinement)
            # ========================================
            
            config.verbose && println("  Iter $iter: iLQGames")
            
            ξ_new, converged = ilqgames_step!(ξ, prob, config)
            
            if !converged
                # iLQGames diverged, fall back to DFMT*
                config.verbose && println("    iLQGames diverged, falling back to DFMT*")
                use_ilqgames = false
                continue
            end
            
            ξ = ξ_new
            
        else
            # ========================================
            # PHASE 1: DFMT* (Global Exploration)
            # ========================================
            
            config.verbose && println("  Iter $iter: DFMT*")
            
            # Adaptive sampling budget
            n_samples = iter == 1 ? config.n_samples_init : config.n_samples_refine
            σ_exploit = 0.3 / (1 + iter)  # Tighten over iterations
            
            ξ_new = dfmt_best_response!(ξ, prob, config, n_samples, σ_exploit)
            ξ = ξ_new
        end
        
        # Compute VI residual
        residual = compute_vi_residual(ξ, prob)
        push!(jaguar.diagnostics.residuals, residual)
        
        config.verbose && println("    Residual: $(round(residual, digits=4))")
        
        # Early termination
        if residual < config.ε_nash
            config.verbose && println("  ✓ Converged to Nash equilibrium")
            break
        end
    end
    
    # Store diagnostics
    push!(jaguar.diagnostics.iteration_counts, iter)
    push!(jaguar.diagnostics.phase_switches, phase_switch_iter)
    
    # Verify safety
    if !verify_safety(ξ, prob)
        @warn "Solution violates safety constraints, attempting emergency replan"
        ξ = emergency_replan!(jaguar)
    end
    
    return ξ
end


