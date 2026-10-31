# CCiLQGames

**Chance-Constrained iterative LQ Stochastic Games.** Locally stationary feedback Nash policy of a nonlinear stochastic game with Gaussian-quantile surrogates of private and shared chance constraints, after Zhong, Shimizu and Chen [ZhongShimizuChen2023](@cite).

## 1. Overview and intended use

Use `CCiLQGames` for a `StochasticGameProblem` with nonlinear dynamics, Gaussian noise and constraints that must hold with a given probability, e.g. keep a position bound with probability ``\ge 0.95``. It returns a feedback policy, the covariance trajectories, per-event risk allocations and their guarantee class, and a Monte Carlo violation summary.

## 2. Supported problem and equilibrium

- Problem: `StochasticGameProblem` over a `GameProblem` with `ItoSDEDynamics`; chance constraints are declared with `chance_constraints = [...]` (private or shared) and optionally a trajectory-wide `joint_risk = JointRiskBudget(ε)`. Ordinary inequality constraints in the nominal game are enforced as deterministic residuals.
- Information: `PerfectStateInformation` or `SharedObservation` (common measurements, EKF with gains frozen along the nominal). Private/asymmetric information is rejected.
- Equilibrium: a *local* stationary point of the Gaussian-surrogate game (`:FeedbackNash`).
- Capabilities: `[:Stochastic, :ChanceConstrained, :FeedbackPolicies, :DiscreteTime, :NonlinearGame, :LQGame]`.

## 3. Algorithm

Outer loop: inequality augmented Lagrangian with multiplier ``\nu^i_{k,m}\ge0`` and penalty ``\rho^i_{k,m}`` per event, updated by ``\nu^+=[\nu+\rho c]_+``. Shared constraints carry one *common* multiplier for all players involved. For a chance event

```math
c_{k,m}=g_{k,m}(\bar\psi_k)+\Phi^{-1}(1-\varepsilon_{k,m})\sqrt{\nabla g^\top\Sigma^\psi_k\nabla g},\qquad \psi=(x,u),
```

with ``\Sigma^\psi_k`` the covariance of the *physical* state and control under the current policy.

Inner loop: (1) nominal stochastic rollout; (2) Jacobians of the discretized drift; (3) chance residuals; (4) per-player local quadratic game (joint-control Hessians plus Gauss-Newton augmented-Lagrangian terms, projected to positive semidefinite); (5) feedback Nash solve by [`FNELQ`](fnelq.md); (6) damped, bounded-step update; stopped when ``\max\|\alpha\|_\infty\le`` `stationarity_tol`.

## 4. Assumptions and guarantees

| Statement | Strength |
|---|---|
| Each inner step solves a local LQ game exactly | exact |
| Chance residual for an *affine* ``g`` under Gaussian law | exact (`:exact_affine_gaussian`) |
| Chance residual for nonlinear ``g`` | first-order approximation (`:linearized_approximation`) |
| Joint failure probability ``\le\varepsilon_{\mathrm{total}}`` for `JointRiskBudget` | union bound; valid only if every individual statement is valid |
| Convergence of the nonlinear game to a global equilibrium | **not claimed** |

`covariance_mode = :frozen_covariance` (the only implemented mode) holds the covariance-dependent term and drops the expected-cost trace term ``\tfrac12\operatorname{tr}(Q\Sigma)`` inside each local game, as in the fast mode of the paper. Stationarity is therefore that of the frozen-covariance surrogate, not of the full stochastic game; `solver_info[:frozen_terms]` lists what was frozen. `:consistent_covariance` raises an error.

Constraints are imposed at stages ``k=1,\dots,N`` (not at the terminal state).

## 5. Minimal example

```@example ccilqgames
using DifferentialGames, LinearAlgebra

dt, N = 0.1, 10
A = Matrix(1.0I(2)); B = [reshape([dt, 0.0], 2, 1), reshape([0.0, dt], 2, 1)]
Q1 = [4.0 0.0; 0.0 0.0]; Q2 = [0.0 0.0; 0.0 4.0]
g = LQGameProblem(A, B, [Q1, Q2], [fill(0.1, 1, 1), fill(0.1, 1, 1)], [Q1, Q2], [0.0, 0.0], N * dt;
                  dt = dt, q = [-(Q1 * [1.0, 0.0]), -(Q2 * [0.0, 1.0])])

# Player 1 must keep x₁ ≤ 0.6 with probability ≥ 0.9 (the bound has two rows, so ε/2 each).
cc = ChanceConstraint(StateBounds(1; state_offset = 0, state_dim = 1, lower = [-5.0], upper = [0.6]), 0.1)
sg = StochasticGameProblem(g, ProcessNoiseModel(Matrix(0.01I(2))); chance_constraints = [cc])

sol  = solve(sg, CCiLQGames(mc_samples = 5000, seed = 1))
info = sol.solver_info
(status = info[:status], feasibility = info[:r_feas], guarantee = info[:guarantee_classes])
```

Per-event probabilities versus the trajectory-wide probability:

```@example ccilqgames
ub = filter(c -> c.component == 2, info[:chance_checks])     # the upper-bound row
(largest_event_probability = maximum(c.violation_probability for c in ub),
 any_violation_monte_carlo = info[:monte_carlo].any_violation,
 union_bound = info[:analytic_joint_violation_bound])
```

A per-event risk of 5 % does **not** give a 95 % trajectory-wide guarantee. Ask for the joint guarantee with a budget:

```@example ccilqgames
budget = JointRiskBudget(0.1)
sg2  = StochasticGameProblem(g, ProcessNoiseModel(Matrix(0.01I(2)));
                             chance_constraints = [cc], joint_risk = budget)
sol2 = solve(sg2, CCiLQGames(mc_samples = 5000, seed = 1))
(joint_bound = sol2.solver_info[:joint_risk_bound],
 any_violation_monte_carlo = sol2.solver_info[:monte_carlo].any_violation)
```

## 6. Options

| Option | Default | Meaning |
|---|---|---|
| `max_iter`, `max_inner_iter` | `20`, `30` | outer (multiplier) and inner (iLQ) iterations |
| `constraint_tol`, `stationarity_tol` | `1e-4`, `1e-3` | feasibility of the surrogate residuals; ``\max\|\alpha\|_\infty`` |
| `ρ0`, `ρ_growth`, `ρ_max` | `10`, `5`, `1e4` | augmented-Lagrangian penalty schedule (very large caps degrade conditioning) |
| `regularization` | `1e-6` | added to each player's own-control Hessian block |
| `covariance_mode` | `:frozen_covariance` | `:consistent_covariance` is reserved |
| `evaluator` | `GaussianQuantileEvaluator()` | chance residual |
| `line_search_β`, `η_min`, `max_state_step` | `0.5`, `1e-3`, `1e3` | bounded backtracking |
| `mc_samples`, `seed` | `2000`, `0` | Monte Carlo validation (`0` disables) |

## 7. Convergence checks and interpretation

`solver_info[:status]` is one of `:locally_stationary_surrogate`, `:surrogate_feasible` (feasible but not stationary), `:max_iterations`, `:line_search_failed`, `:singular_nash_system`, `:ill_conditioned_nash_system`, `:second_order_violation`. `sol.converged` is `true` for the first two. Three quantities are tracked separately: stationarity `:r_stat`, surrogate feasibility `:r_feas`, and per-iteration step size (in `:residual_history`). A small step alone is not convergence. `:chance_checks` give each event's residual, exact or approximate violation probability and guarantee class; `:monte_carlo` is an *estimate* (Wilson intervals), not a guarantee, and a zero count is not a safety proof.

## 8. Known limitations

- Frozen-covariance mode only; no consistent-covariance derivative.
- No scalar merit exists for general-sum games: the line search only bounds the step and damps while stationarity stalls.
- Terminal-state chance constraints are not imposed.
- Nonlinear separation under shared observations is an EKF/certainty-equivalence approximation.
- Very large penalties (``\rho\gtrsim10^6``) produced oscillating inner iterations in the 3-player test.

## 9. Complexity

Per inner iteration: Jacobians and covariance propagation ``O(N n^3)``, local-game assembly with automatic differentiation of ``(n+m)\times(n+m)`` Hessians per player and stage, and one FNELQ solve ``O(N m^3)`` (not benchmarked).

## 10. References

[ZhongShimizuChen2023](@cite) (algorithm and Gaussian chance surrogate); [BasarOlsder1999](@cite). The implementation differs from the paper in the damped bounded-step update and in solving each local game with FNELQ.
