# SFNELQ

**Stochastic Feedback Nash Equilibrium, Linear-Quadratic.** Feedback Nash equilibrium of an LQ game with *additive* process noise.

## 1. Overview and intended use

Use `SFNELQ` when a `StochasticGameProblem` wraps a deterministic LQ game (`LQGameProblem`/`LTVLQGameProblem`) and the noise enters through a constant diffusion (additive noise). It returns the feedback Nash strategies together with each player's *expected* cost.

## 2. Supported problem and equilibrium

- Problem: `StochasticGameProblem` whose `nominal_game` is an LQ game and for which `is_certainty_equivalent(sgame)` is `true`.
- Equilibrium: feedback Nash (`equilibrium_type = :FeedbackNash`) in expectation.
- Capabilities: `[:LQGame, :Stochastic, :FeedbackPolicies, :DiscreteTime]`.

## 3. Algorithm

For additive zero-mean noise with covariance ``W``, **certainty equivalence** holds: the feedback gains equal those of the deterministic game, so `SFNELQ` calls [`FNELQ`](fnelq.md) and adds the expected noise cost

```math
\Delta J_i = \sum_{k=1}^{N} \tfrac12 \operatorname{tr}\big(Z_{i,k+1} W\big),
```

where ``Z_{i,k}`` are the cost-to-go matrices of the deterministic recursion ([Optimality Conditions](../math/optimality_conditions.md)).

## 4. Assumptions and guarantees

Everything FNELQ assumes (non-singular ``S_k``), plus zero-mean additive noise. State- or control-multiplicative noise is **not** supported: `solve` throws an `ErrorException` for a `StochasticGameProblem` with state/control-dependent diffusion.

## 5. Minimal example

```@example sfnelq
using DifferentialGames, LinearAlgebra

A = [1.0 1.0; 0.0 1.0]
B = [reshape([0.0, 1.0], 2, 1)]
g = LQGameProblem(A, B, [Matrix(1.0I(2))], [fill(0.1, 1, 1)], [Matrix(10.0I(2))], [1.0, 0.0], 2.0; dt = 1.0)

sgame = StochasticGameProblem(g, ProcessNoiseModel(Matrix(0.01I(2))))
sol   = solve(sgame, SFNELQ())
det   = solve(g, FNELQ())

(noise_cost = sol.solver_info[:noise_cost],
 expected_cost = get_cost(sol, 1),
 deterministic_cost = get_cost(det, 1))
```

```@example sfnelq
# Certainty equivalence: identical gains.
get_gain(get_strategy(sol), 1, 1) ≈ get_gain(get_strategy(det), 1, 1)
```

## 6. Options

Same as FNELQ: `check_singularity` (`true`), `rcond_threshold` (`1e-10`), `regularization` (`0.0`).

## 7. Convergence checks and interpretation

`sol.converged` reports the backward pass (conditioning of ``S_k``). The reported cost of each player is the **expected** cost: deterministic cost plus `sol.solver_info[:noise_cost][i]` (always ≥ 0 in the tests). `sol.solver_info[:certainty_equivalent]` is `true`.

## 8. Known limitations

- Additive noise only (see §4).
- Gains do not depend on the noise level (a property of certainty equivalence, not a limitation of the code); the strategy therefore does not hedge against risk.
- No sampled-trajectory simulation is provided; `state_trajectory` is the mean trajectory.

## 9. Complexity

Same as FNELQ plus ``O(N n^3)`` for the trace terms (not benchmarked).

## 10. References

[BasarOlsder1999](@cite), Ch. 6.
