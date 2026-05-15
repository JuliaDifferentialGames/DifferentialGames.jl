# Nonlinear Games

Most real-world problems have nonlinear dynamics — unicycle vehicles, spacecraft with attitude, robotic arms. The **iLQGames** solver handles these by iteratively approximating the game as a sequence of LQ problems around the current trajectory.

## Background: iLQGames

iLQGames (iterative Linear-Quadratic Games) works as follows:

1. **Initialize** a nominal trajectory (e.g., zero controls)
2. **Linearize** the dynamics around the current trajectory: ``f(x,u) \approx A_k x + B_k u + c_k``
3. **Quadraticize** each player's cost: ``\ell_i(x,u) \approx x^\top Q_{i,k} x + u^\top R_{i,k} u + \ldots``
4. **Solve** the resulting LQ game with FNELQ to get updated feedback strategies
5. **Rollout** the new strategies with a line search to ensure cost decrease
6. **Repeat** until convergence (``\|\Delta \text{trajectory}\|_\infty < \varepsilon``)

This converges to a **local** feedback Nash equilibrium. The quality of the solution depends on initialization and problem structure.

## Example: Two Unicycles at an Intersection

A common benchmark from the autonomous-driving literature: two unicycle vehicles approach a narrow passage.

```julia
using DifferentialGames, LinearAlgebra

# ── Unicycle dynamics ─────────────────────────────────────────
# State:   [x, y, v, θ]  (position, speed, heading)
# Control: [a, ω]         (acceleration, angular rate)

function unicycle(xi, ui, p, t)
    x, y, v, θ = xi[1], xi[2], xi[3], xi[4]
    a, ω       = ui[1], ui[2]
    return [v * cos(θ); v * sin(θ); a; ω]
end

# ── Costs ─────────────────────────────────────────────────────
# Both players want to reach a goal, penalize speed and input effort.
# Q penalizes [x, y, v, θ], R penalizes [a, ω]

stage    = DiagonalLQStageCost([0.1, 0.1, 1.0, 0.1], [0.1, 0.1])
terminal = DiagonalLQTerminalCost([10.0, 10.0, 1.0, 0.1])

# ── Collision avoidance ───────────────────────────────────────
# Shared constraint: vehicles must stay > 1.5 m apart
collision = SharedInequality([1, 2];
    func = (x, u, p, t) -> [1.5^2 - ((x[1]-x[5])^2 + (x[2]-x[6])^2)],
    dim  = 1
)

# ── Players ───────────────────────────────────────────────────
# Player 1 approaches from the left, player 2 from the bottom
player1 = PlayerSpec(1, 4, 2, [-3.0, 0.0, 1.0, 0.0],   unicycle,
                     PlayerObjective(1, stage, terminal))
player2 = PlayerSpec(2, 4, 2, [ 0.0, -3.0, 1.0, π/2],  unicycle,
                     PlayerObjective(2, stage, terminal))

# ── Game ──────────────────────────────────────────────────────
game = PDGNEProblem([player1, player2], [collision], 4.0, 0.1)

# ── Solve ─────────────────────────────────────────────────────
sol = solve(game, iLQGames(; max_iter=100, ε_conv=1e-4, verbose=true))

println("converged:     ", sol.converged)
println("iterations:    ", sol.iterations)
println("Player 1 cost: ", round(get_cost(sol, 1); digits=4))
println("Player 2 cost: ", round(get_cost(sol, 2); digits=4))
```

## Working with Nonlinear Costs

Costs do not need to be quadratic. Use `NonlinearStageCost` and `NonlinearTerminalCost` for any differentiable function:

```julia
# Player 1 wants to stay close to a reference trajectory (e.g., a lane center)
ref_lane = [0.0, 0.0, 1.0, 0.0]   # desired state

lane_cost = NonlinearStageCost(
    (x, u, p, t) -> begin
        Δx = x - ref_lane
        return 2.0 * (Δx[1]^2 + Δx[2]^2) + 0.1 * u' * u
    end;
    is_separable = true
)

terminal_cost = NonlinearTerminalCost(
    x -> 10.0 * ((x[1] - ref_lane[1])^2 + (x[2] - ref_lane[2])^2)
)

obj = PlayerObjective(1, lane_cost, terminal_cost)
```

!!! note "ForwardDiff compatibility"
    iLQGames uses `ForwardDiff.jl` to compute Jacobians and Hessians of dynamics and costs. Your functions must be compatible with dual-number arithmetic — avoid `if`-branches on state values or inplace mutations of arrays.

## Coupled Nonlinear Dynamics

When player states are coupled (e.g., rigid-body contact), use `CoupledNonlinearDynamics` with the joint state vector ``x = [x_1; x_2; \ldots]``:

```julia
# Joint state: [x1, y1, x2, y2], joint control: [u1x, u1y, u2x, u2y]
function coupled_dyn(x, u, p, t)
    # Simple coupling: player 2's dynamics affected by player 1's position
    dx1 = u[1:2]
    dx2 = u[3:4] + 0.1 * [x[1] - x[3]; x[2] - x[4]]  # attraction term
    return [dx1; dx2]
end

dynamics = CoupledNonlinearDynamics(coupled_dyn, 4, 4)  # n=4, m_total=4

# Build game directly (no PlayerSpec for coupled dynamics)
# ... (see PDGNEProblem reference for the full constructor)
```

## Convergence and Initialization

iLQGames finds a **local** Nash equilibrium. The solution depends on initialization:

```julia
# Default: zero-control rollout from x0 (built in)
sol = solve(game, iLQGames())

# The solver accepts warmstart data for re-solving from a prior solution:
warmstart = WarmstartData(sol)
sol2 = solve(game_perturbed, iLQGames(); warmstart=warmstart)
```

**Tips for convergence:**
- Reduce the horizon or timestep if the solver diverges
- Scale costs so no one term dominates by orders of magnitude (use `diagnose_scaling`)
- If constraint violation is large, try ALGAMES instead (see [Constrained Games](constrained_games.md))

## Further Reading

- [iLQGames solver reference](../solvers/ilqgames.md) — full option list
- [Constrained Games tutorial](constrained_games.md) — adding hard constraints with ALGAMES
- [`PDGNEProblem` reference](../problem_types/pdgnep.md) — building separable-dynamics games
