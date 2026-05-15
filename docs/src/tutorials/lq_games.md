# LQ Games

Linear-quadratic (LQ) differential games have linear dynamics and quadratic costs. They admit closed-form feedback Nash equilibria via backward Riccati recursion, making them an important special case: exact, fast, and analytically tractable.

## Background

In a discrete-time, finite-horizon N-player LQ game, each player ``i`` solves:

```math
\min_{u_i} \sum_{k=0}^{N-1} \left( x_k^\top Q_i x_k + u_{i,k}^\top R_i u_{i,k} \right) + x_N^\top Q_{f,i} x_N
```

subject to the shared linear dynamics:

```math
x_{k+1} = A x_k + \sum_j B_j u_{j,k}
```

The **feedback Nash equilibrium** consists of time-varying gain matrices ``\{K_{i,k}\}`` such that each player's strategy ``u_{i,k} = -K_{i,k} x_k`` simultaneously satisfies all players' optimality conditions. These are computed by solving a system of coupled backward Riccati equations.

## Shared-State LQ Game (LQGameProblem)

Use `LQGameProblem` when all players share a single joint state vector (the most common academic formulation).

```julia
using DifferentialGames, LinearAlgebra

# ── Problem parameters ────────────────────────────────────────
n  = 4   # state dimension (shared)
m  = 1   # control dimension per player
N  = 2   # number of players
T  = 3.0 # horizon (seconds)
dt = 0.1 # timestep

# Discrete-time dynamics: x_{k+1} = A x_k + B1 u1 + B2 u2
A  = [1.0 dt 0.0 0.0;
      0.0 1.0 0.0 0.0;
      0.0 0.0 1.0 dt;
      0.0 0.0 0.0 1.0]

B  = [reshape([0.0, dt, 0.0, 0.0], n, 1),   # player 1 acts on x-velocity
      reshape([0.0, 0.0, 0.0, dt], n, 1)]    # player 2 acts on y-velocity

# Per-player cost matrices
Q  = [diagm([1.0, 0.0, 0.0, 0.0]),   # player 1 cares about x-position
      diagm([0.0, 0.0, 1.0, 0.0])]   # player 2 cares about y-position
R  = [fill(0.01, m, m), fill(0.01, m, m)]
Qf = [10.0 * Q[1], 10.0 * Q[2]]

x0 = [2.0, 0.0, 2.0, 0.0]   # initial state

game = LQGameProblem(A, B, Q, R, Qf, x0, T; dt=dt)

# ── Solve ─────────────────────────────────────────────────────
sol = solve(game, FNELQ())

println("converged: ", sol.converged)
println("J₁ = ", round(get_cost(sol, 1); digits=4))
println("J₂ = ", round(get_cost(sol, 2); digits=4))

# ── Extract feedback gains ────────────────────────────────────
gains = sol.solver_info[:feedback_gains]
# gains[i][k] is the (m×n) gain matrix for player i at time step k
println("K₁ at t=0: ", gains[1][1])
```

## Separable LQ Game (PDGNEProblem with LQ costs)

When each player has its own independent state, use `PDGNEProblem` (Partially-Decoupled GNEP). This is the typical setup for multi-robot scenarios.

```julia
using DifferentialGames, LinearAlgebra

# Two double-integrators in 1D
dyn = (xi, ui, p, t) -> [xi[2]; ui[1]]   # ẋ = v, v̇ = u

make_player(id, x0) = PlayerSpec(
    id, 2, 1, x0, dyn,
    PlayerObjective(id,
        DiagonalLQStageCost([1.0, 0.1], [0.01]),   # penalize position + velocity
        DiagonalLQTerminalCost([10.0, 1.0])          # terminal position penalty
    )
)

player1 = make_player(1, [2.0, 0.0])
player2 = make_player(2, [-2.0, 0.0])

game = PDGNEProblem([player1, player2], 3.0, 0.1)
sol  = solve(game, iLQGames())    # iLQGames converges exactly on LQ problems

println("J₁ = ", get_cost(sol, 1))
println("J₂ = ", get_cost(sol, 2))
```

## Time-Varying LQ Game (LTVLQGameProblem)

For problems where the dynamics or costs change over time:

```julia
N   = 30                              # number of time steps
A_seq = [I(4) for _ in 1:N]          # constant A for this example
B_seq = [[I(4)[:, 1:1]] for _ in 1:N],   # one B per player per step
        [[I(4)[:, 2:2]] for _ in 1:N]]
Q_seq = [[diagm(ones(4))] for _ in 1:N],
        [[diagm(ones(4))] for _ in 1:N]]
R_seq = [[fill(0.1, 1, 1)] for _ in 1:N],
        [[fill(0.1, 1, 1)] for _ in 1:N]]
Qf    = [diagm(ones(4)), diagm(ones(4))]
x0    = [1.0, 0.0, -1.0, 0.0]

# game = LTVLQGameProblem(A_seq, B_seq, Q_seq, R_seq, Qf, x0, 3.0; dt=0.1)
```

## Accessing Feedback Gains

FNELQ returns the time-varying gain matrices alongside the trajectories:

```julia
sol = solve(game, FNELQ())

gains = sol.solver_info[:feedback_gains]
# gains is a Vector{Vector{Matrix}} — gains[player][timestep]

# Player 1's gain at the final step before terminal
K1_final = gains[1][end]

# Reconstruct the closed-loop policy:
#   u₁(k) = -K1[k] * x(k)
```

## Single-Player Limit (LQR)

A one-player game is classical LQR. FNELQ handles this correctly:

```julia
n, m = 4, 2
A  = I(n)
B  = [I(n)[:, 1:m]]             # single player with m controls
Q  = [diagm(ones(n))]
R  = [0.1 * I(m)]
Qf = [10.0 * diagm(ones(n))]
x0 = ones(n)

lqr_game = LQGameProblem(A, B, Q, R, Qf, x0, 2.0; dt=0.1)
sol = solve(lqr_game, FNELQ())
```

## Further Reading

- [FNELQ solver reference](../solvers/fnelq.md) — solver options and implementation notes
- [LQGameProblem reference](../problem_types/lq_game.md) — full constructor API
- [Nonlinear Games tutorial](nonlinear_games.md) — extend to nonlinear dynamics with iLQGames
