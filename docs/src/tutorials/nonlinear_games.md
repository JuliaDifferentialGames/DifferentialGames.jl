# Nonlinear Games

Most real-world problems have nonlinear dynamics — unicycle vehicles, spacecraft with attitude, robotic arms. The **iLQGames** solver handles these by iteratively approximating the game as a sequence of LQ problems around the current trajectory.

## Background: iLQGames

iLQGames (iterative Linear-Quadratic Games) works as follows:

1. **Initialize** a nominal trajectory (zero controls from ``x_0``).
2. **Linearize** the dynamics around the current trajectory: ``f(x,u) \approx A_k x + B_k u + c_k``.
3. **Quadraticize** each player's cost along the trajectory.
4. **Solve** the resulting LQ game (FNELQ) for updated feedback strategies.
5. **Roll out** the new strategies with a line search.
6. **Repeat** until the trajectory change falls below `ε_conv`.

If it terminates, the result is a **local** feedback-Nash *approximation*: the stopping test is on the change in consecutive trajectories, which is not by itself a proof of equilibrium. iLQGames does **not** support constraints; use [ALGAMES](constrained_games.md) for those.

## Example: Two Unicycles

```@example nl
using DifferentialGames, LinearAlgebra

# State: [x, y, v, θ] (position, speed, heading); control: [a, ω]
function unicycle(xi, ui, p, t)
    v, θ = xi[3], xi[4]
    return [v * cos(θ); v * sin(θ); ui[1]; ui[2]]
end

stage    = DiagonalLQStageCost([0.1, 0.1, 1.0, 0.1], [0.1, 0.1])
terminal = DiagonalLQTerminalCost([10.0, 10.0, 1.0, 0.1])

player1 = PlayerSpec(1, 4, 2, [-3.0,  0.0, 1.0, 0.0],  unicycle, PlayerObjective(1, stage, terminal))
player2 = PlayerSpec(2, 4, 2, [ 0.0, -3.0, 1.0, π/2], unicycle, PlayerObjective(2, stage, terminal))

game = PDGNEProblem([player1, player2], 4.0, 0.1)

sol = solve(game, iLQGames(; max_iter=100))
(converged = sol.converged, iterations = sol.iterations,
 J = round.(get_costs(sol); digits=3))
```

Inspect the convergence history — a decreasing sequence of trajectory changes is the evidence for convergence, not the boolean alone:

```@example nl
Δ = sol.solver_info[:trajectory_change_history]
println("trajectory change: first = ", round(first(Δ); digits=3),
        ", last = ", round(last(Δ); digits=4), "  (threshold ε_conv = 0.05)")
println("cost history: ", round.(sol.solver_info[:cost_history]; digits=2))
```

## Working with Nonlinear Costs

Costs need not be quadratic. `NonlinearStageCost` takes `(x, u, p, t) -> scalar` and `NonlinearTerminalCost` takes `(x, p) -> scalar`, where `x` is the **joint** state (all players concatenated). Both must be ForwardDiff-compatible.

```@example nl
ref = [0.0, 0.0]          # lane centre (position only)

lane_cost = NonlinearStageCost(
    (x, u, p, t) -> 2.0 * ((x[1] - ref[1])^2 + (x[2] - ref[2])^2) + 0.1 * sum(abs2, u);
    is_separable = true,
)
goal_cost = NonlinearTerminalCost((x, p) -> 10.0 * ((x[1] - ref[1])^2 + (x[2] - ref[2])^2))

obj = PlayerObjective(1, lane_cost, goal_cost)
typeof(obj)
```

!!! note "iLQGames passes each player's own state and control"
    With `iLQGames` a `NonlinearStageCost`/`NonlinearTerminalCost` function receives **player i's own** state `x` (length `nᵢ`) and control `u` (length `mᵢ`), so `x[1]`, `x[2]` are that player's position. `ALGAMES` instead passes the *joint* state and control ([Constrained Games](constrained_games.md)); `x[1]`, `x[2]` would then be player 1's position for every player. See [Conventions](../ecosystem/conventions.md) for a portable idiom.

!!! note "ForwardDiff compatibility"
    iLQGames uses `ForwardDiff.jl` for Jacobians and Hessians. Avoid in-place mutation of arrays and branches on state values.

## Convergence and Initialization

iLQGames finds a **local** equilibrium; it can be sensitive to the horizon, the initial state and the cost scaling. Practical advice:

- Inspect `sol.converged`, `sol.iterations` and `sol.solver_info[:trajectory_change_history]`.
- If the solver stops at `max_iter` without converging, its result is an approximation; do not treat it as an equilibrium.
- `iLQGames(; max_state_step=...)` limits how far the state may move in one line-search step, and `μ_init` sets the initial regularization of the LQ solve; both affect how many iterations are needed.
- Use `diagnose_scaling(game)` to look for cost terms that differ by orders of magnitude.
- Warm-start with `solve(game2, iLQGames(); warmstart=sol)`.

## Further Reading

- [iLQGames solver reference](../solvers/ilqgames.md) — full option list
- [Constrained Games tutorial](constrained_games.md) — adding hard constraints with ALGAMES
- [`PDGNEProblem` reference](../problem_types/pdgnep.md) — building separable-dynamics games
