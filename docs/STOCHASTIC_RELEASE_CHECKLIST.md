# Stochastic games release checklist

Work lives on branch `stochastic-games-overhaul` in `DifferentialGamesBase.jl`,
`DifferentialGamesBaseSolvers.jl`, `DifferentialGamesBenchmarks.jl` and this repository.

## Merge order (the packages pin each other through `[sources]` on `main`)

1. **DifferentialGamesBase.jl** — merge first; nothing downstream builds against the new API until `main` has it
   (`JointLQStageCost`, `StochasticGameProblem` fields, risk budgets, Kalman helpers, `:ChanceConstrained`).
2. **DifferentialGamesBaseSolvers.jl** — then merge (`CCiLQGames`, SFNELQ/FNELQ changes; adds `Random` as a dependency).
3. **DifferentialGamesBenchmarks.jl** — then merge (`StochasticLQ`, `ChanceConstrainedLQ` families; adds `Random`).
4. **DifferentialGames.jl** — last (docs, umbrella tests). The docs build executes `@example` blocks against Base and
   BaseSolvers `main`, so it fails before steps 1–2 land.

For local work before merging, `Pkg.develop` the sibling checkouts in a scratch environment instead of editing `[sources]`.

## Gate before tagging

- [ ] Base: `Pkg.test()` green on Julia 1.10 (declared minimum) — only 1.12 was run during development.
- [ ] BaseSolvers: full `Pkg.test()` green. Development ran FNELQ, SFNELQ, CCiLQGames, RHN, iLQGames, ALGAMES and the
      regression/capability tests; DGSQP, LIBR, YIPAVEL, INVLQ, FALCON and DTRHN suites were not re-run after the FNELQ rewrite.
- [ ] Benchmarks: full `Pkg.test()` green (development ran the core, LQ-formation and stochastic suites).
- [ ] Umbrella: `Pkg.test()` green; `julia --project=docs docs/make.jl` builds with `warnonly = false`.
- [ ] `julia --project=. benchmark/stochastic_scaling.jl` (BaseSolvers) — compare against the previous release's CSV.
- [ ] Version bumps and `[compat]` entries for the new `Random` dependencies; decide whether the FNELQ `converged`
      semantics change (now also `false` for a violated second-order condition) needs a minor-version bump.

## Behaviour changes to call out in the release notes

- `FNELQ`: fixed the missing `-Mα` term when cross terms and affine terms are both active; time-varying costs are now
  detected with fixed dynamics; singular / ill-conditioned / non-Nash stage games return `converged = false` with
  `solver_info[:status]` instead of throwing or warning only.
- `ProcessNoiseModel` accepts positive-semidefinite (including zero) covariances and per-step sequences;
  `StochasticGameProblem` gained `initial_covariance`, `information`, `chance_constraints`, `risk_allocation`, `joint_risk`.
- `GaussianQuantileEvaluator` quantile is now accurate to ~1e-16 (was ~2e-5 in the tails).
- `determinize` adds the `½tr(QΣ)` term for quadratic costs and starts from `initial_covariance`.
- `SFNELQ` noise cost respects objective `scaling`.

## Documented limits (do not over-claim)

- `CCiLQGames`: frozen-covariance mode only; local, not global, equilibria; chance constraints at stages `1…N`;
  per-event risks do not bound trajectory-wide failure (use `JointRiskBudget`); large penalties degrade convergence.
- Exact probability statements hold for affine constraints under a Gaussian law; nonlinear ones are first-order.
- Shared observations: separation is exact for linear-Gaussian common-information LQ games only; private or asymmetric
  observations are rejected.
- Not implemented: consistent-covariance derivatives, multiplicative noise in the LQ solver, distributionally robust
  chance constraints, optimized risk allocation, MC/scenario/unscented evaluators (interfaces reserved).
