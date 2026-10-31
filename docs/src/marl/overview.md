# DifferentialGamesMARL.jl

`DifferentialGamesMARL.jl` brings **multi-agent reinforcement learning** to the ecosystem. Where the solvers in `DifferentialGamesBaseSolvers.jl` use the game's model (dynamics, costs, derivatives), the learners here only *interact* with the game as an environment and learn neural-network policies from experience.

!!! warning "Work in progress"
    The package was migrated from exploratory work and its API may change. Its test suite covers environment construction, observations, constraint modes, obstacles, the replay buffer and a single network update; the training loop and the guided-policy-search bridge are **not** covered by tests. The examples in this section were run against the current sources but are **not executed** during the documentation build (they need Flux and take seconds to minutes).

## What it provides

| Component | Page | Purpose |
|-----------|------|---------|
| `MADDPGEnv` | [Environments](environments.md) | wraps any Base `GameProblem` as a `CommonRLInterface.AbstractEnv`: reward = −cost, with swappable constraint handling |
| `cooperative_navigation`, `predator_prey`, polygon obstacles | [Environments](environments.md) | benchmark tasks built as ordinary Base games |
| `MADDPGSolver`, `Agent`, `ReplayBuffer`, `train!` | [MADDPG](maddpg.md) | the multi-agent actor-critic learner ([Lowe2017](@cite)) |
| `build_guidance`, `gps_builder`, `LocalGameSolver` | [Guided policy search](gps.md) | pulls actor training towards a *model-based* local LQ-game solution computed with `FNELQ` |

## How it connects to the rest of the ecosystem

| Step | Package | Object |
|------|---------|--------|
| 1. describe the game | Base | `GameProblem` |
| 2. wrap it as an environment | MARL | `MADDPGEnv(game)` |
| 3. train | MARL | `train!(MADDPGSolver(…), env)` → trained `Agent`s |
| 3′. (optional) guide training | MARL → Solvers → Base | `gps_builder` → `FNELQ` on a local LQ model (`expand` → `assemble_lq_game` → solve) → `GuidanceLaw` |

- **Problems come from Base.** A learning problem *is* a Base game: `MADDPGEnv(game)` reads the dynamics, objectives, constraints, initial state and horizon from the same `GameProblem` you would pass to `solve`. The built-in tasks return ordinary games, so a model-based solver can be run on the very same task for comparison.
- **The learner is not a `GameSolver`.** `MADDPGSolver` is a hyperparameter struct trained with `train!(solver, env)`; there is no `solve` method and no `GNEPSolution`. Learned agents are evaluated by rolling them out in the environment.
- **Solvers come in through guidance only.** `gps_builder` calls the `FNELQ` solver on a local LQ approximation of the game (obtained with Base's `expand`/`assemble_lq_game`) and turns it into a guidance controller for the actor loss.
- **Stochastic problems live upstream.** Problem definitions with noise, chance constraints and the POMDPs.jl adapter are in Base; `StochasticLQNash` (local guidance with multiplicative noise) is not ported yet and raises an error. Additive-noise guidance can be built with `SFNELQ`.

## Installation

The package is not registered. Add it by URL together with the packages it builds on (see [Installation](../getting_started/installation.md)):

```julia
using Pkg
Pkg.add([
    PackageSpec(url="https://github.com/JuliaDifferentialGames/DifferentialGamesBase.jl"),
    PackageSpec(url="https://github.com/JuliaDifferentialGames/DifferentialGamesBaseSolvers.jl"),
    PackageSpec(url="https://github.com/JuliaDifferentialGames/DifferentialGamesMARL.jl"),
])
Pkg.add("CommonRLInterface")      # for reset!, act!, observe, terminated
```

`DifferentialGamesMARL` imports `CommonRLInterface` but does not reexport its functions; add it to your environment to step an environment yourself.

## Which approach should I use?

| If you have… | Prefer |
|--------------|--------|
| a model with derivatives and a local equilibrium is enough | [`iLQGames`](../solvers/ilqgames.md) / [`ALGAMES`](../solvers/algames.md) |
| a model but non-smooth or very nonlinear costs, and a distribution of start states | MADDPG with a *distribution* of starts (`reset_fn`) |
| a model and a need for global-ish behavior, but want to keep model-based structure | MADDPG with [guided policy search](gps.md) |
| only a simulator | MADDPG (the environment never needs derivatives) |

A learned policy is a neural network, not an equilibrium: nothing in `train!` certifies a Nash equilibrium. The `track_best` option returns the best-scoring policy by greedy evaluation, which is the wrong notion for adversarial games (see below).
