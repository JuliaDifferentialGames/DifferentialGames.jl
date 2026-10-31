# Cross-Package Conventions

Because the packages are developed together but implement different methods, a few conventions differ between solvers. This page lists them in one place. Each entry was checked against the source or by running the code; entries marked *(inconsistency)* are known mismatches that are tracked in the audit.

## States and trajectories of a `PDGNEProblem`

A `PDGNEProblem` has one state block per player; the **joint state** stacks the blocks (`x = [x₁; x₂; …]`, offsets in `game.metadata.state_offsets`).

| Solver | `get_trajectory(sol, i).states` | `sol.state_trajectory` |
|--------|----------------------------------|------------------------|
| `FNELQ` (shared-state LQ games) | shared state `n × (N+1)` | the same matrix |
| `iLQGames`, `ALGAMES` | **joint** state `n × (N+1)`, identical for every player | set (joint) |
| `LIBR` | player `i`'s **own** state `nᵢ × (N+1)` | `nothing` |
| `YiPavel` | player `i`'s decision, as a one-step "trajectory" | see its page |

*(inconsistency)* The [`GNEPSolution`](../basics/solutions.md) docstring states that `state_trajectory` is `nothing` for PD-GNEPs, where each player has private state; `iLQGames` and `ALGAMES` set it. Code that reads trajectories generically (plotting, evaluators, receding-horizon shifts) must handle both layouts. The Plots helpers resolve positions from `game.metadata`; the receding-horizon warm-start shift handles both layouts.

## What a hand-written cost function receives

For `NonlinearStageCost(f)` with `f(x, u, p, t)` and `NonlinearTerminalCost(g)` with `g(x, p)` in a `PDGNEProblem` (checked by instrumenting the functions):

| Solver | `x` | `u` (stage cost) |
|--------|-----|------------------|
| `ALGAMES` | the **joint** state (length `Σnᵢ`) | the **joint** control (length `Σmᵢ`) |
| `iLQGames` (and Base's `expand`) | **player i's own** state (length `nᵢ`) | **player i's own** control (length `mᵢ`) |
| `DifferentialGamesMARL` `MADDPGEnv` | joint (`cost_state = :joint`, default) or own (`:player`) | own |

*(inconsistency)* A function written for one convention errors (`BoundsError`) or silently computes a different cost under the other; for example `0.01 * sum(abs2, u)` charges a player for *all* players' control effort under `ALGAMES` but only its own under `iLQGames`. `DiagonalLQStageCost`, `LQStageCost` and the cost-term DSL handle the offsets themselves. Shared constraints are evaluated on the joint state (only `ALGAMES` supports them).

A portable way to write a nonlinear cost for player `i` (state block `o_x+1 : o_x+nᵢ`, control block `o_u+1 : o_u+mᵢ`) is to select the own block only when a joint vector is passed:

```julia
own_state(x)   = length(x) == n_total ? x[o_x+1 : o_x+n_i] : x
own_control(u) = length(u) == m_total ? u[o_u+1 : o_u+m_i] : u
```

This was verified on a two-player game solved with both `ALGAMES` and `iLQGames`.

In `DifferentialGamesMARL`, `MADDPGEnv(…; cost_state = :joint)` (default) passes the joint state to a stage cost, matching the built-in tasks; `cost_state = :player` passes the player's slice.

## Time discretization

| Where | Scheme |
|-------|--------|
| `LQGameProblem`, `LTVLQGameProblem` | already discrete: you supply `A`, `B` |
| `iLQGames` linearization (`expand`) | zero-order hold (`ZOHDiscretization()`) by default |
| rollouts of continuous dynamics (`iLQGames` line search, `ALGAMES`, `rollout`) | RK4 with the control held constant |
| `LIBR` | forward Euler |
| `DifferentialGamesMARL` `MADDPGEnv` | forward Euler (or, with `discrete_dynamics = true`, the dynamics are treated as a map); the GPS bridge linearizes the *same* Euler map (`euler_da`) |

Mixed schemes are consistent for linear dynamics with a constant control over a step, and differ at higher order for nonlinear dynamics. Compare solvers on a nonlinear problem only with that in mind.

## Costs

- Reported costs of LQ objectives include a factor ½: ``J_i=\tfrac12[\sum_k(x^\top Qx+u^\top Ru)+x_N^\top Q_fx_N]`` (checked against the LQR Riccati solution).
- In `MADDPGEnv` the reward is the negative cost, with constraint terms added through the chosen `constraint_mode` (`:barrier`, `:penalty` or `:none`); rewards are therefore not directly comparable with `GNEPSolution` costs.

## Strategies

- `FeedbackStrategy`: ``u_k^i=\hat u_k^i-P_k^i(x_k-\hat x_k)-\eta\,\alpha_k^i`` (`apply_strategy`). `get_gain` returns ``P_k^i``. After the fix recorded in the audit, `FNELQ`'s feedforward uses this sign convention consistently.
- `OpenLoopStrategy`: a control sequence per player; `ALGAMES` and `LIBR` return this.

## Constraint handling across solvers

| Constraint | iLQGames | ALGAMES | MADDPG env | YiPavel |
|------------|----------|---------|------------|---------|
| control/state bounds | no | yes (nodes `1:N`) | `action_limit`, barrier/penalty, optional projection | convex private sets |
| shared inequality | no | yes (nodes `1:N`; `terminal_constraints=true` adds `x(N+1)`) | barrier/penalty in the reward | affine coupling |

## Capabilities

`solver_capabilities(S)` and `required_capabilities(game)` draw from the closed set `CANONICAL_CAPABILITIES`. `solve` warns when a game requires a capability the solver does not declare, but a declared capability is a hint, not a guarantee; read the solver's *Known limitations*.

## Where the ecosystem draws the line on `solve`

`solve(game, solver; warmstart, verbose, check_compatibility)` applies to every `GameSolver`. The MARL package's `MADDPGSolver` is **not** a `GameSolver`; it is trained with `train!`. `RecedingHorizonNash` *is* a `GameSolver` and wraps another one. `InverseLQGames` is a `GameSolver` whose problem is an `InverseGameProblem` and whose result is a set of cost parameters rather than an equilibrium.
