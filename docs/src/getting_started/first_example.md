# First Example

This page walks through a complete two-player differential game from problem specification to solution extraction. Two unicycle-model vehicles start apart and each drives toward the origin region while keeping speed and control effort low.

## The Problem

Each vehicle has state ``[x, y, v, \theta]`` (position, speed, heading) and controls ``[a, \omega]`` (acceleration, turn rate):

```math
\dot x = v\cos\theta, \quad \dot y = v\sin\theta, \quad \dot v = a, \quad \dot\theta = \omega .
```

Player ``i`` minimizes its own quadratic cost

```math
J_i = \tfrac12\Big[\sum_{k=0}^{N-1} x_{i,k}^\top Q_i x_{i,k} + u_{i,k}^\top R_i u_{i,k} + x_{i,N}^\top Q_{f,i} x_{i,N}\Big]
```

The factor ``\tfrac12`` is the convention used by the solvers' reported costs (see [Working with Solutions](../basics/solutions.md)). In this example the players do not interact through their costs, so the equilibrium is simply each player's own optimal-control solution; see the [Constrained Games tutorial](../tutorials/constrained_games.md) for genuine coupling.

## Code

```@example first
using DifferentialGames, LinearAlgebra

# Per-player continuous-time dynamics ẋ = f(xᵢ, uᵢ, p, t).
# Dynamics are SEPARABLE: players interact only through costs and shared constraints.
unicycle(xi, ui, p, t) = [xi[3]*cos(xi[4]); xi[3]*sin(xi[4]); ui[1]; ui[2]]

stage    = DiagonalLQStageCost([0.1, 0.1, 1.0, 0.1], [0.1, 0.1])
terminal = DiagonalLQTerminalCost([10.0, 10.0, 1.0, 0.1])

player1 = PlayerSpec(1, 4, 2, [-3.0,  0.0, 1.0, 0.0],  unicycle, PlayerObjective(1, stage, terminal))
player2 = PlayerSpec(2, 4, 2, [ 0.0, -3.0, 1.0, π/2], unicycle, PlayerObjective(2, stage, terminal))

game = PDGNEProblem([player1, player2], 4.0, 0.1)

# iLQGames: local feedback-Nash approximation for unconstrained nonlinear games.
sol = solve(game, iLQGames())
(converged = sol.converged, iterations = sol.iterations)
```

!!! note "Which state do cost functions see?"
    In a `PDGNEProblem` each player's *dynamics* act on that player's own state.
    `DiagonalLQStageCost` and other LQ costs are applied to each player's own block automatically.
    Custom `NonlinearStageCost`/`NonlinearTerminalCost` functions are solver dependent:
    `ALGAMES` evaluates them on the **joint** state `[x₁; x₂; …]` and joint control, `iLQGames` on the
    player's own slices (see [Conventions](../ecosystem/conventions.md) and the
    [Constrained Games](../tutorials/constrained_games.md) tutorial).

## Accessing the Solution

```@example first
traj1 = get_trajectory(sol, 1)

# For PD-GNEP solutions `states` holds the JOINT state, 8 × (N+1): player 1 occupies
# rows 1:4 and player 2 rows 5:8. `controls` holds that player's own controls, 2 × N.
@show size(traj1.states) size(traj1.controls)

println("Final position of player 1: ", round.(traj1.states[1:2, end]; digits=2))
println("Final position of player 2: ", round.(traj1.states[5:6, end]; digits=2))
println("costs: ", round.(get_costs(sol); digits=3))
println("has strategy: ", has_strategy(sol), "  (", typeof(get_strategy(sol)), ")")
```

`sol.converged` reports the solver's own stopping test, not a proof that the
point is a Nash equilibrium. See [Working with Solutions](../basics/solutions.md).

## Using a Different Solver

Because all solvers share the same `solve` interface, switching is one word.
`FNELQ` is exact but only applies to games built with `LQGameProblem` /
`LTVLQGameProblem` (linear dynamics on a shared state):

```@example first
lq_game = LQGameProblem(
    Matrix(1.0I(4)), [Matrix(1.0I(4))[:, 1:2], Matrix(1.0I(4))[:, 3:4]],
    [Matrix(1.0I(4)), Matrix(1.0I(4))],
    [Matrix(0.1I(2)), Matrix(0.1I(2))],
    [Matrix(10.0I(4)), Matrix(10.0I(4))],
    [3.0, 0.0, -3.0, 0.0], 5.0; dt=0.1
)

sol_fnelq = solve(lq_game, FNELQ())
sol_fnelq.converged
```

## Next Steps

- **[LQ Games tutorial](../tutorials/lq_games.md)** — full treatment of the linear-quadratic case
- **[Nonlinear Games tutorial](../tutorials/nonlinear_games.md)** — unicycle vehicles and iLQGames
- **[Building Problems](../basics/problem_interface.md)** — detailed guide to `PlayerSpec`, costs, constraints
