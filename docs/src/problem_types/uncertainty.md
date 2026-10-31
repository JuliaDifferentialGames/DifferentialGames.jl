# Stochastic, Robust and Partially Observable Games

Three problem types layer uncertainty on top of a *nominal* deterministic game. Each **wraps** the nominal game (`nominal_game` or `underlying_game`) instead of copying it, so everything that works on the nominal game still applies to it.

| Type | Uncertainty | Wrapper arguments | Consumed by |
|------|-------------|-------------------|-------------|
| `StochasticGameProblem` | random process noise (Itô diffusion or a covariance) | nominal game + `ProcessNoiseModel` or an `ItoSDEDynamics` | [`SFNELQ`](../solvers/sfnelq.md) (additive noise); `determinize` |
| `RobustGameProblem` | bounded, non-probabilistic disturbance | nominal game + `DisturbanceSet(w_min, w_max)` | tube-based solvers such as [`DynamicTubeMPC`](../solvers/dtmpc.md) (the tube parameters carry the disturbance bound), the [Benchmarks](../ecosystem/benchmarks.md) robustness challenge |
| `POGameProblem` | partial or noisy observation | underlying game + one observation model per player | information-structure queries; no built-in forward solver |

`required_capabilities` adds `:Stochastic`, `:RobustDisturbance` or `:PartiallyObservable` to those of the nominal game, so `solve`'s capability check reports a solver that does not handle the extension.

## Stochastic games

```@docs
StochasticGameProblem
ProcessNoiseModel
```

With *additive* noise the problem is **certainty equivalent**: the feedback Nash gains equal those of the deterministic game and the noise only adds an expected-cost term. `SFNELQ` exploits this. Multiplicative (state- or control-dependent) noise is representable through `ItoSDEDynamics` but no solver here handles it.

```@example uncertainty
using DifferentialGames, LinearAlgebra

A = [1.0 0.1; 0.0 1.0]
B = [reshape([0.0, 0.1], 2, 1)]
nominal = LQGameProblem(A, B, [Matrix(1.0I(2))], [fill(0.1, 1, 1)], [Matrix(5.0I(2))], [1.0, 0.0], 1.0; dt = 0.1)

stochastic = StochasticGameProblem(nominal, ProcessNoiseModel(Matrix(0.01I(2))))
(certainty_equivalent = is_certainty_equivalent(stochastic),
 multiplicative_noise = has_multiplicative_noise(stochastic),
 required = required_capabilities(stochastic))
```

```@example uncertainty
sol_det = solve(nominal, FNELQ())
sol_sto = solve(stochastic, SFNELQ())
(deterministic_cost = get_cost(sol_det, 1), expected_cost = get_cost(sol_sto, 1),
 noise_cost = sol_sto.solver_info[:noise_cost][1])
```

The expected cost exceeds the deterministic one by exactly `noise_cost` (checked in the solver's tests).

### Determinization and chance constraints

`determinize(stochastic; propagator, evaluator, risk_allocation)` compiles a stochastic game into a deterministic `GameProblem` whose state is the belief ``z=(\mu,\Sigma)`` (the mean and the flattened covariance), propagating the covariance with a `LinearizedGaussianPropagation` (or an unscented or particle propagator). A `ChanceConstraint(constraint, risk)` states ``\Pr[g(x,u)\le 0]\ge 1-\varepsilon`` and is compiled into a deterministic residual on the belief by `compile_chance_constraint` with a chance evaluator (Gaussian quantile, Monte Carlo or scenario) and a risk allocation (uniform, fixed or optimized).

```@example uncertainty
cc = ChanceConstraint(PrivateInequality(1; func = (x, u, p, t) -> [x[1] - 2.0], dim = 1), 0.05)
determinized = determinize(stochastic)
(chance_constraint = cc, belief_state_dim = state_dim(determinized))
```

!!! warning "Current limitation"
    The determinized game's dynamics is a **discrete-time map** wrapped in `CoupledNonlinearDynamics`, which the solvers integrate as a *continuous-time* right-hand side. As a result `solve(determinize(sg), iLQGames())` currently fails (an `InexactError` while indexing the time step), contrary to the `determinize` docstring. The construction itself works; solving it needs a discrete-time nonlinear dynamics type, tracked in the audit.

For solving stochastic games end to end — risk budgets, shared observations, and what each guarantee means — see [How to solve a stochastic game](../tutorials/stochastic_games.md). The complete stochastic API (SDE dynamics, discretization, propagators, chance evaluators, the POMDPs.jl adapter) is documented in the `DifferentialGamesBase.jl` API reference.

## Robust games

```@docs
DisturbanceSet
RobustGameProblem
```

```@example uncertainty
robust = RobustGameProblem(nominal, DisturbanceSet([-0.05, -0.05], [0.05, 0.05]))
(robust, required_capabilities(robust))
```

A robust game describes *worst-case* rather than distributional uncertainty. The set is what tube-based and min-max methods need: the disturbance bound that sizes the tube.

## Partially observable games

```@docs
POGameProblem
```

Each player observes ``y=h(x,v,p,t)`` through a `PrivateObservation` (own signal; strategies depend on beliefs) or a `SharedObservation` (a common signal), defined in [Information Structures](../math/information_structures.md):

```@example uncertainty
position_only = (x, v, p, t) -> x[1:1]                           # observe the first state, noisily
po = POGameProblem(nominal, [PrivateObservation(position_only, 1, fill(0.01, 1, 1))])
(po, required_capabilities(po))
```

No solver in this ecosystem consumes a `POGameProblem` directly. The type fixes the interface so that belief-space solvers, and the `restrict_observation` challenge of the Benchmarks package, can be written against it.
