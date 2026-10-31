# How to solve a stochastic game

This guide covers Gaussian LQ games ([`SFNELQ`](../solvers/sfnelq.md)) and nonlinear games with chance constraints ([`CCiLQGames`](../solvers/ccilqgames.md)), and states precisely what each result does and does not guarantee.

## Which solver, which information

| Problem | Solver | Information supported |
|---|---|---|
| Linear dynamics, quadratic costs, additive Gaussian noise (time-varying, possibly singular), uncertain initial state | `SFNELQ` | perfect state; *common* affine measurements |
| Nonlinear dynamics, Gaussian noise, private/shared chance constraints | `CCiLQGames` | perfect state; *common* measurements (EKF) |
| Private or asymmetric observations | none | rejected with an error: separation of estimation and control is not valid |
| Multiplicative (state/control-dependent) noise in an LQ game | none | `SFNELQ` raises an error |

## Build the problem

A stochastic game wraps an ordinary game; nothing is duplicated.

```julia
sgame = StochasticGameProblem(game, ProcessNoiseModel(W);          # or DiscreteGaussianNoise
                              initial_covariance = Σ0,
                              information = PerfectStateInformation(),
                              chance_constraints = [cc],
                              joint_risk = JointRiskBudget(0.1))
```

- `W` is one positive-semidefinite matrix or one per step. Zero and rank-deficient covariances are allowed.
- For a nonlinear game pass an `ItoSDEDynamics` instead of a `ProcessNoiseModel`.
- Player costs may depend on every player's control through `JointLQGameProblem` (full ``R_i`` of size ``m\times m``). Only `FNELQ`, `SFNELQ` and `CCiLQGames` consume joint-control costs.

## Chance constraints and risk

`ChanceConstraint(constraint, ε)` states ``\Pr[g(x,u)\le0]\ge1-\varepsilon``. Private constraints enter one player's cost; shared constraints enter every player involved with one common multiplier. A constraint with several output rows (a two-sided bound has two) splits ``\varepsilon`` equally across rows.

Every event ``(c,k,m)`` has its own risk ``\varepsilon_{c,k,m}``. By the union bound

```math
\sum_{c,k,m}\varepsilon_{c,k,m}\le\varepsilon_{\mathrm{total}}\ \Longrightarrow\ \Pr[\text{all constraints hold at all steps}]\ge1-\varepsilon_{\mathrm{total}},
```

**provided each individual statement is valid** — exact for affine constraints under a Gaussian law, only approximate for nonlinear ones. `JointRiskBudget(ε_total)` allocates uniformly over all events; inspect the allocation with `allocate_horizon_risk` and `check_risk_budget`.

Without a budget, per-event risks do not compose: in the [`CCiLQGames` example](../solvers/ccilqgames.md) a 5 % per-event risk leaves roughly a one-in-three chance of at least one violation along the trajectory.

## Check a solution without trusting the solver

`check_chance_constraints` evaluates any chance constraint on Gaussian state/control moments, and `empirical_violation_rates` summarises sampled constraint values:

```julia
checks = check_chance_constraints(ccs, X̄, Ū, Σx, Σu, Σxu)
[(c.step, c.violation_probability, c.guarantee) for c in checks]
```

Each `ChanceCheck` reports whether the probability is exact (`:exact_affine_gaussian`) or a linearized approximation. Simulation estimates carry sampling error and a zero count is not a guarantee.

## What the results mean

| Quantity | Meaning |
|---|---|
| `:state_covariances` | covariance of the **physical** state under the policy |
| `:posterior_covariances` | the filter's estimation covariance (shared observation only) — not the same thing |
| `:r_feas` (CCiLQGames) | worst positive surrogate residual |
| `:r_stat` (CCiLQGames) | stationarity of the *frozen-covariance* local game, not of the full stochastic game |
| `:status` | `:locally_stationary_surrogate`, `:surrogate_feasible`, … see the solver pages |
