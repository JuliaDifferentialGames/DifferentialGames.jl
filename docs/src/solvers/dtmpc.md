# DynamicTubeMPC

**Per-agent dynamic-tube model predictive control** wrapped around a constrained game solver. It produces a receding-horizon plan for a game in which each agent faces a *bounded disturbance*, and it co-optimizes a scalar *tube width* per agent inside the game so that coupled (for example collision) constraints are tightened by the size of the tube.

!!! warning "Research-grade solver"
    `DynamicTubeMPC` implements a recent research formulation. Its source header, the theory audit in the repository (`THEORY_AUDIT.md`) and its tests are the authoritative references. This page summarizes the interface and states what is verified here; its examples are **not executed** in this manual (the solve takes tens of seconds, and the reference inner solver FALCON is an external, unregistered package).

## 1. Overview and intended use

For a `PDGNEProblem` horizon game, at each receding-horizon step the solver builds an **augmented** game and hands it to an inner game solver. Per agent ``i`` and tube channel ``k`` (translational, optionally rotational and detection) it adds a width state ``s`` and a bandwidth state ``\alpha`` with its own artificial control ``v``:

```math
\dot s = -\alpha s + \beta,\qquad \dot\alpha = v,\qquad \alpha\in[\alpha_{lo},\alpha_{\max}],\ |v|\le v_{\max},
```

(discretized exactly as ``s_{t+1}=e^{-\alpha_t\Delta t}s_t+\tfrac{\beta}{\alpha_t}(1-e^{-\alpha_t\Delta t})``). The augmented per-agent state is ``\tilde x_i=[x_i;\ s_1\dots s_{n_{ch}};\ \alpha_1\dots\alpha_{n_{ch}}]`` and the control ``\tilde u_i=[u_i;\ v_1\dots v_{n_{ch}}]``. The original cost is left **unchanged** (no extra cost curvature on the tube states). Coupled constraints are tightened by the tube widths, for example the collision constraint ``d_{\min}-\lVert r_i-r_j\rVert + s_i + s_j \le 0``.

At execution time an ancillary boundary-layer sliding controller keeps the true state inside the tube around the planned nominal.

## 2. Supported problem and equilibrium

- Problem: `DynamicTubeMPCProblem(game, x0, n_sim_steps, params, uncertainty_models)` where `game` is a `PDGNEProblem` (separable dynamics, one state block per agent). A shared-state game such as an `LQGameProblem` is rejected.
- Inner solver: any constrained `GameSolver`; validated in the repository with FALCON (external). `ALGAMES` can be used (see limitations).
- Solution: `DynamicTubeMPCSolution` with the closed-loop trajectory `X`, nominal controls `U`, per-agent `costs`, tube width trajectories `s_trajectories` and bandwidths.

## 3. Algorithm

The tube structure follows *Dynamic Tube MPC for Nonlinear Systems* ([Lopez2019](@cite)); the tube **bandwidth** is a co-optimized state following the Outland–Arya formulation referenced in the source header, which is authoritative where it differs from the earlier design note. Initial-node freedom (the nominal initial state is a decision variable constrained by initial containment) requires an inner solver that supports freeing the initial-state pin; solvers without that notion (such as `ALGAMES`) keep the classical hard pin, a valid but more conservative special case.

## 4. Assumptions and guarantees

- The original costs are time-invariant (LQ in the repository's validation) and the tube-bandwidth lower bound `α_lo` is positive; the width recursion divides by ``\alpha`` and is unstable for ``\alpha\le0``.
- The tube hedges against a bounded disturbance from *nature*. It does **not** claim feedback/Markov-perfect Nash optimality, time consistency of the open-loop equilibrium, or robustness to a strategic opponent.
- The verification status of each theoretical claim is tracked in `src/solvers/DTRHN/THEORY_AUDIT.md` of the solvers repository.

## 5. Minimal example (not executed)

This example ran successfully against the local sources with `ALGAMES` as the inner solver (two agents on a line, about 19 s); the closed-loop solve converged and the coupled collision constraint was satisfied. It uses discrete-time per-agent dynamics.

```julia
using DifferentialGames
using DifferentialGamesBaseSolvers: create_independent_uncertainty_models

dt = 0.1; N_win = 6
di(xi, ui, p, t) = [xi[1] + dt * xi[2], xi[2] + dt * ui[1]]             # discrete double integrator
stage(x, u, p, t) = 1.0 * x[1]^2 + 0.1 * x[2]^2 + 0.5 * u[1]^2
term(x, p)        = 5.0 * x[1]^2 + 1.0 * x[2]^2
players = [PlayerSpec(i, 2, 1, [(i == 1 ? -1.0 : 1.0), 0.0], di,
                      PlayerObjective(i, NonlinearStageCost(stage), NonlinearTerminalCost(term)))
           for i in 1:2]
game = PDGNEProblem(players, N_win * dt, dt)                              # horizon game

params = DTMPCParameters(Float64, 2;
    s₀ = fill(0.1, 2), α_fixed = fill(1.0, 2), D = fill(0.05, 2), η = fill(0.01, 2),
    coupled_pairs = [(1, 2)], collision_min_distance = 0.5, pos_dim = 1)
uncertainty = create_independent_uncertainty_models(x -> 0.05, 2)

prob = DynamicTubeMPCProblem(game, [-1.0, 0.0, 1.0, 0.0], 6, params, uncertainty)
sol  = solve(prob, DynamicTubeMPC(ALGAMES(; tol_opt = 1e-2); warm_start = false))

dtmpc_tube_trajectory(sol, 1)        # tube width of agent 1 over the simulation
get_total_cost(sol, 1)
all_coupled_constraints_satisfied(sol)
```

## 6. Configuration

`DTMPCParameters(Float64, n_agents; kwargs...)` takes per-agent vectors `s₀` (initial width), `α_fixed` (initial bandwidth seed), `D` (disturbance bound), `η` (robustness margin), `pos_dim`, optional `coupled_pairs` with `collision_min_distance`, and options for rotational and detection channels, actuator/rate tightening, `α_lo`, `α_max`, `v_max`, `boundary_layer` gains and `κ_speed`. `DynamicTubeMPC(inner_solver; warm_start, replan_stride, …)` wraps the inner solver. See the docstrings in `src/solvers/DTRHN/src/dtmpc.jl` for the complete list.

!!! note
    The standalone `DTMPC.md` in the solvers repository still shows an older constructor (`α₀`, `w_Φ`, `create_independent_uncertainty_models` only). The example above uses the *current* keywords, checked by running it.

## 7. Convergence checks and interpretation

`sol.converged` reports the closed-loop solve; the accessors `dtmpc_tube_trajectory`, `dtmpc_bandwidth_trajectory`, `get_state_trajectory`, `get_control_trajectory`, `get_total_cost` and `all_coupled_constraints_satisfied` give what to inspect. Tube widths grow over the horizon when the disturbance accumulates; coupled constraints must hold with the tightening already included.

## 8. Known limitations

- **ALGAMES and warm starts.** `ALGAMES` enforces shared inequality constraints only approximately, so the exact coupled tube-allocation equilibrium is not reproduced; the mechanism is wired and its first-order effect verified in the repository's theory validation.
- With `ALGAMES` use `warm_start = false`. The warm-start shift assumes each player's `Trajectory` holds the *augmented private* state (FALCON's convention) while `ALGAMES` stores the joint state in every trajectory (see [Conventions](../ecosystem/conventions.md)); with `warm_start = true` the tail correction fails with a dimension error.
- The actuator-reserve tightening ``\alpha s`` is bilinear and non-convex when ``\alpha`` is co-optimized, which moves those constraints to the penalized path.

## 9. Complexity

Each receding-horizon step solves an augmented game with ``2n_{ch}`` extra states and ``n_{ch}`` extra controls per agent; cost is that of the inner solver on the augmented problem (not benchmarked here).

## 10. References

[Lopez2019](@cite) for the dynamic tube; the formulation and its audit are in the solvers repository (`src/solvers/DTRHN/`).
