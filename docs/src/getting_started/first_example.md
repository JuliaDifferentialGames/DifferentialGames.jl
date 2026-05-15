# First Example

This page walks through a complete two-player differential game from problem specification to solution extraction. We use a **two-spacecraft rendezvous** scenario: both vehicles try to reach a common target while managing fuel cost.

## The Problem

Two spacecraft in 2D move under double-integrator dynamics:

```math
\dot{x}_i = v_i, \quad \dot{v}_i = u_i, \quad i = 1, 2
```

After discretization at ``\Delta t = 0.1`` s each player solves:

```math
\min_{u_i} \; \sum_{k=0}^{N-1} \left[ x_i^\top Q_i x_i + u_i^\top R_i u_i \right] + x_i(N)^\top Q_f x_i(N)
```

subject to the other player also playing optimally. The Nash equilibrium is the solution where neither player can improve their cost by changing their control, given the other's strategy.

## Code

```julia
using DifferentialGames, LinearAlgebra

# ── Dynamics ──────────────────────────────────────────────────
# State: [x, y, vx, vy] for each player (total 8 states)
# Each player controls [ax, ay] (2 actuators)
#
# Separable dynamics: each player has independent state

dyn = (xi, ui, p, t) -> [xi[3]; xi[4]; ui[1]; ui[2]]   # ẋ=v, v̇=u

# ── Costs ─────────────────────────────────────────────────────
stage    = DiagonalLQStageCost([0.0, 0.0, 1.0, 1.0], [0.1, 0.1])
terminal = DiagonalLQTerminalCost([10.0, 10.0, 1.0, 1.0])

# ── Players ───────────────────────────────────────────────────
player1 = PlayerSpec(1, 4, 2, [3.0, 0.0, 0.0, 0.0], dyn,
                     PlayerObjective(1, stage, terminal))

player2 = PlayerSpec(2, 4, 2, [-3.0, 0.0, 0.0, 0.0], dyn,
                     PlayerObjective(2, stage, terminal))

# ── Game ──────────────────────────────────────────────────────
game = PDGNEProblem([player1, player2], 5.0, 0.1)

# ── Solve ─────────────────────────────────────────────────────
sol = solve(game, iLQGames(); verbose=true)

println("converged:      ", sol.converged)
println("Player 1 cost:  ", round(get_cost(sol, 1); digits=4))
println("Player 2 cost:  ", round(get_cost(sol, 2); digits=4))
```

## Accessing the Solution

```julia
# Retrieve the trajectory for player 1
traj1 = get_trajectory(sol, 1)

# traj1.states  is (4 × N+1)  — state at each time step
# traj1.controls is (2 × N)   — control at each time step
# traj1.cost    is a Float64  — total cost for player 1

println("Final position (P1): ", traj1.states[1:2, end])
println("Final position (P2): ", get_trajectory(sol, 2).states[1:2, end])

# The shared strategy (feedback gains) if available
if has_strategy(sol)
    strat = get_strategy(sol)
end
```

## Using a Different Solver

Because all solvers share the same `solve` interface, switching is one word:

```julia
# Exact LQ feedback Nash (only valid for LQ problems)
lq_game = LQGameProblem(
    I(4), [I(4)[:, 1:2], I(4)[:, 3:4]],
    [diagm(ones(4)), diagm(ones(4))],
    [0.1*I(2), 0.1*I(2)],
    [10.0*diagm(ones(4)), 10.0*diagm(ones(4))],
    [3.0, 0.0, -3.0, 0.0], 5.0; dt=0.1
)

sol_fnelq = solve(lq_game, FNELQ())
```

## Next Steps

- **[LQ Games tutorial](../tutorials/lq_games.md)** — full treatment of the linear-quadratic case
- **[Nonlinear Games tutorial](../tutorials/nonlinear_games.md)** — unicycle vehicles and iLQGames
- **[Building Problems](../basics/problem_interface.md)** — detailed guide to `PlayerSpec`, costs, constraints
