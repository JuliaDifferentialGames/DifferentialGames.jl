# Guided Policy Search

**Guided policy search (GPS)** adds a penalty to the actor loss that pulls each actor towards the control a *local model-based solution* would apply. In this package the local solution is the feedback Nash equilibrium of an LQ approximation of the game, computed with [`FNELQ`](../solvers/fnelq.md) from `DifferentialGamesBaseSolvers.jl`. It is the point where the model-based and learning routes of the ecosystem meet.

## Idea

Every `guide_every` episodes the training loop

1. rolls the current greedy policy out to get a nominal trajectory `(X, U)` (`greedy_nominal`);
2. linearizes the game's dynamics and quadraticizes its costs around it (Base's `expand`), choosing the discretization that matches the environment's own map (`euler_da` for forward Euler, none for a discrete-time `LinearDynamics` game);
3. assembles the LTV LQ game (`assemble_lq_game`) and solves it with a `LocalGameSolver`;
4. wraps the result in a `GuidanceLaw`: per-player affine feedback `u = u*(t) − K(t)(x − x̄(t))` in absolute coordinates, sampled at each nominal step, clamped to the action range.

With `ρ > 0` the actor loss gets the guidance term; `ρ = 0` (default) skips it entirely.

## Local solvers

| Type | Meaning |
|------|---------|
| `DeterministicLQNash()` | coupled Riccati recursion with no noise terms: plain `FNELQ` on the assembled game (default) |
| `SinglePlayerLQR()` | one-player case (classic GPS) |
| `StochasticLQNash(noise)` | **not ported**: constructing it works, solving raises an error. Use an additive-noise `StochasticGameProblem` with [`SFNELQ`](../solvers/sfnelq.md) if noisy local guidance is needed |

## Usage (verified on an LQ-cost game)

```julia
using DifferentialGamesMARL, DifferentialGamesBase, LinearAlgebra, Random

n = 4
game = LQGameProblem(Matrix(0.95I(n)),
        [Matrix(1.0I(n))[:, 1:1], Matrix(1.0I(n))[:, 3:3]],
        [Matrix(1.0I(n)), Matrix(1.0I(n))], [fill(0.1, 1, 1), fill(0.1, 1, 1)],
        [Matrix(10.0I(n)), Matrix(10.0I(n))], [1.0, 0.0, -1.0, 0.0], 2.0; dt = 0.1)
env = MADDPGEnv(game; discrete_dynamics = true, cost_state = :joint, action_limit = 2.0)

solver = MADDPGSolver(; episodes = 15, warmup = 30, batch_size = 16, buffer_size = 1000,
                        eval_every = 5, hidden = 8,
                        ρ = 0.1,                                              # guidance weight
                        guidance_builder = gps_builder(DeterministicLQNash(); clamp_to = (-2.0, 2.0)),
                        guide_every = 5)
agents, history, agent_history, evals = train!(solver, env; rng = MersenneTwister(3))

evals.guide_log     # [(6, "ok"), (11, "ok")]: episode and outcome of each guidance rebuild
```

A guidance law can also be built and inspected directly:

```julia
g = build_guidance(agents, env, DeterministicLQNash(); clamp_to = (-2.0, 2.0))
guidance_diagnostics(g, [env.state])   # (mean_dist, max_dist, saturated_fraction)
```

`guidance_diagnostics` reports how far a batch of states lies from the nominal (a far batch is extrapolating) and how often the guidance control saturates (a control pinned at the limit carries no gradient information beyond "go to the edge").

## Known limitations

- **Built-in nonlinear tasks do not work yet.** On `cooperative_navigation` the builder fails: the task's cost indexes the *joint* state, while Base's `expand` passes a player's *own* state slice to a `NonlinearStageCost` (`BoundsError` in `_stage_hessian`). `gps_builder` catches the error and logs `"builder declined"` in `evals.guide_log`, so training continues **unguided**. GPS currently works for games whose costs are LQ (the example above) or written for the slice convention. See [Conventions](../ecosystem/conventions.md).
- **Not covered by the package's tests**, as the training loop is not.
- Deterministic local solves only; trust-region and outer-loop GPS variants, multiplicative-noise local solves and `local_exploitability` are not ported (`StochasticLQNash` and `local_exploitability` raise errors).
- The guidance is *local*: it is a linearization around the current greedy rollout and is only meaningful near it.

## References

The penalty form follows the guided-policy-search literature cited in `src/solver.jl`; the formulation used there is not reproduced or verified in this manual.
