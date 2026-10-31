# SFNELQ

**Stochastic Feedback Nash Equilibrium, Linear-Quadratic.** Feedback Nash equilibrium of a Gaussian LQ game with *additive* process noise, an uncertain initial state, and (optionally) *common* noisy measurements.

## 1. Overview and intended use

Use `SFNELQ` when a `StochasticGameProblem` wraps a deterministic LQ game (`LQGameProblem`, `LTVLQGameProblem` or `JointLQGameProblem`) and the noise is additive. It returns the feedback Nash strategies, each player's *expected* cost, and the mean and covariance trajectories of the physical state and control.

## 2. Supported problem and equilibrium

- Problem: `StochasticGameProblem` whose `nominal_game` is an LQ game and for which `is_certainty_equivalent(sgame)` is `true`. Process covariances may be time-varying and singular (positive semidefinite); the initial state may be uncertain (`initial_covariance`).
- Costs: own-control (`LQStageCost`) or full joint-control (`JointLQStageCost`) quadratic player costs with affine terms.
- Information: `PerfectStateInformation`, or `SharedObservation` with an *affine common* measurement ``y_k = H x_k + c + v_k``. Private, asymmetric, open-loop and nonlinear observation models are rejected with an `ErrorException`.
- Equilibrium: feedback Nash (`equilibrium_type = :FeedbackNash`) in expectation.
- Capabilities: `[:LQGame, :Stochastic, :FeedbackPolicies, :DiscreteTime]`.

## 3. Algorithm

For additive zero-mean noise, **certainty equivalence** holds: the feedback gains equal those of the deterministic game, so `SFNELQ` calls [`FNELQ`](fnelq.md). With perfect state information the expected cost is the deterministic cost plus

```math
\Delta J_i = \tfrac12\operatorname{tr}(Z_{i,1}\Sigma_0) + \sum_{k=1}^{N} \tfrac12 \operatorname{tr}\big(Z_{i,k+1} W_k\big),
```

scaled by the objective's `scaling`, where ``Z_{i,k}`` are the cost-to-go matrices ([Optimality Conditions](../math/optimality_conditions.md)).

With a `SharedObservation`, the players act on a *common* Kalman estimate ``\hat x_k`` (``u_k=-P_k\hat x_k-\alpha_k``). The first measurement is assimilated before the first control. The solver tracks the joint law of the state ``x_k`` and the estimation error ``e_k=x_k-\hat x_k``; the expected cost is evaluated exactly from those moments:

```math
J_i = J_i^{\mathrm{det}} + \sum_k \tfrac12\operatorname{tr}(Q\,\Sigma^x_k)+\tfrac12\operatorname{tr}(R\,\Sigma^u_k)+\langle M,\operatorname{Cov}(x_k,u_k)\rangle + \tfrac12\operatorname{tr}(Q_f\Sigma^x_{N+1}).
```

## 4. Assumptions and guarantees

- *Proved* (risk-neutral, additive Gaussian noise, nonsingular stage Nash systems [BasarOlsder1999](@cite)): the gains are the deterministic Nash gains and the stated expected costs are exact. With a common affine measurement, separation of estimation and feedback-Nash control holds because all players share the same information.
- *Not covered:* state- or control-multiplicative noise, private/asymmetric measurements, chance constraints (use [`CCiLQGames`](ccilqgames.md)). Each raises an error rather than returning a solution whose guarantee does not apply.
- The physical-state covariance (`:state_covariances`) is **not** the filter's posterior covariance (`:posterior_covariances`): the control depends on the random estimate, so the two differ.

## 5. Minimal example

```@example sfnelq
using DifferentialGames, LinearAlgebra

A = [1.0 0.1; 0.0 1.0]
B = [reshape([0.0, 0.1], 2, 1)]
g = LQGameProblem(A, B, [Matrix(1.0I(2))], [fill(0.5, 1, 1)], [Matrix(3.0I(2))], [1.0, 0.0], 5.0; dt = 1.0)

W     = [Matrix(0.02I(2)) for _ in 1:5]            # time-varying (here constant) covariances
sgame = StochasticGameProblem(g, ProcessNoiseModel(W); initial_covariance = Matrix(0.05I(2)))
sol   = solve(sgame, SFNELQ())
det   = solve(g, FNELQ())

(noise_cost = sol.solver_info[:noise_cost],
 expected_cost = get_expected_costs(sol),
 deterministic_cost = get_cost(det, 1))
```

```@example sfnelq
# Certainty equivalence: identical gains.
get_gain(get_strategy(sol), 1, 1) ≈ get_gain(get_strategy(det), 1, 1)
```

With a noisy common measurement of the first state only:

```@example sfnelq
obs = linear_observation([1.0 0.0], fill(0.05, 1, 1))
sh  = solve(StochasticGameProblem(g, ProcessNoiseModel(W);
                initial_covariance = Matrix(0.05I(2)), information = obs), SFNELQ())

(physical = tr(sh.solver_info[:state_covariances][end]),
 posterior = tr(sh.solver_info[:posterior_covariances][end]),
 perfect_information = tr(get_covariance_trajectory(sol)[end]))
```

## 6. Options

Same as FNELQ: `check_singularity` (`true`), `rcond_threshold` (`1e-10`), `regularization` (`0.0`).

## 7. Convergence checks and interpretation

`sol.converged` reports the backward pass; `sol.solver_info[:status]` is one of `:solved`, `:ill_conditioned_nash_system`, `:second_order_violation`, `:singular_nash_system`. A failed stage game returns zero policies and `converged = false`, never a false equilibrium. Other entries:

| `solver_info` key | Meaning |
|---|---|
| `:expected_costs`, `:noise_cost` | expected cost per player; excess over the deterministic cost |
| `:state_covariances` | ``\operatorname{Cov}(x_k)``, ``k=1,\dots,N+1`` |
| `:control_covariances`, `:state_control_covariances` | ``\operatorname{Cov}(u_k)``, ``\operatorname{Cov}(x_k,u_k)`` |
| `:estimate_covariances`, `:posterior_covariances`, `:kalman_gains` | shared-observation only |
| `:second_order_ok`, `:min_own_hessian_eig` | each player's own-control Hessian block is positive definite |

## 8. Known limitations

- Additive noise only; chance constraints are rejected.
- Gains do not depend on the noise level (a property of certainty equivalence): the strategy does not hedge against risk.
- `state_trajectory` is the mean trajectory; no sampled trajectories are produced.

## 9. Complexity

Same as FNELQ plus ``O(N n^3)`` for the trace and covariance terms (not benchmarked).

## 10. References

[BasarOlsder1999](@cite), Ch. 6.
