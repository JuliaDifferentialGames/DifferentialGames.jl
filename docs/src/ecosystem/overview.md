# The Ecosystem

DifferentialGames.jl is a family of five packages. This page explains what each one is for, how they depend on each other, and how a problem flows from a model to a result, a learned policy or a benchmark score.

## The packages

| Package | Role | Depends on | Provides |
|---------|------|------------|----------|
| **DifferentialGamesBase.jl** | *Language of the ecosystem*: problem types, dynamics, costs, constraints, solutions, and the solver contract | — (ForwardDiff, LinearAlgebra, SparseArrays) | `GameProblem` and the [other problem types](../problem_types/overview.md), `PlayerSpec`, `GNEPSolution`, `GameSolver`, `solve`, `solver_capabilities`, uncertainty propagation and chance constraints |
| **DifferentialGamesBaseSolvers.jl** | *Model-based solvers* | Base | [`FNELQ`](../solvers/fnelq.md), [`SFNELQ`](../solvers/sfnelq.md), [`iLQGames`](../solvers/ilqgames.md), [`ALGAMES`](../solvers/algames.md), [`LIBR`](../solvers/libr.md), [`YiPavel`](../solvers/yipavel.md), [`InverseLQGames`](../solvers/invlq.md), [`RecedingHorizonNash`](../solvers/rhn.md), [`DynamicTubeMPC`](../solvers/dtmpc.md), [`DGSQP`](../solvers/dgsqp.md); a Plots extension and an extension for the soft dependency [`FALCON`](../solvers/falcon.md) |
| **DifferentialGames.jl** (this package) | *Umbrella and documentation*: one `using` for Base + Solvers | Base, Solvers | Reexports both; this manual |
| **DifferentialGamesMARL.jl** | *Learning-based solvers*: multi-agent reinforcement learning | Base, Solvers (+ Flux, Optimisers, CommonRLInterface) | [`MADDPGEnv`](../marl/environments.md), [`MADDPGSolver`](../marl/maddpg.md), [guided policy search](../marl/gps.md), benchmark tasks |
| **DifferentialGamesBenchmarks.jl** | *Evaluation*: benchmark cases, challenge transformations, independent metrics | Base (Solvers and Makie as extensions) | [`BenchmarkCase`s, `Experiment`, evaluators and plots](../ecosystem/benchmarks.md) |

None of the five packages is in the General registry yet, so they are installed by URL (see [Installation](../getting_started/installation.md)). `using DifferentialGames` loads only Base and Solvers; MARL and Benchmarks are separate installs because they bring heavy dependencies (Flux, Makie).

## Dependency structure

Arrows read "depends on".

| Package | Depends on |
|---------|------------|
| `DifferentialGames` | `DifferentialGamesBase`, `DifferentialGamesBaseSolvers` |
| `DifferentialGamesBaseSolvers` | `DifferentialGamesBase` |
| `DifferentialGamesMARL` | `DifferentialGamesBase`, `DifferentialGamesBaseSolvers` (used for the local `FNELQ` step) |
| `DifferentialGamesBenchmarks` | `DifferentialGamesBase`; `DifferentialGamesBaseSolvers` and Makie only as extensions |

Everything depends on **Base**. A new solver package needs *only* Base; see [Writing a solver](../api/solver_interface.md).

## One game, several routes

All routes start from a game object built from Base types. What differs is what consumes it.

| Route | Entry point | Returns | Package |
|-------|-------------|---------|---------|
| Model-based | `solve(game, solver)` with `FNELQ`, `iLQGames`, `ALGAMES`, … | `GNEPSolution` | Solvers |
| Learning-based | `train!(MADDPGSolver(…), MADDPGEnv(game))` | trained agents and learning histories | MARL |
| Learning guided by a model | the same, with `gps_builder(DeterministicLQNash())` | trained agents | MARL, calling `FNELQ` from Solvers |
| Evaluation | `evaluate(evaluator, game(case), sol)` for any solution | validity, quality, robustness and efficiency metrics | Benchmarks |

Two contracts hold the ecosystem together:

1. **The solver contract (Base).** A solver is a subtype of `GameSolver` with `solver_capabilities` and a `_solve(game, solver, warmstart, verbose)` method. Anything that follows it works with `solve`, with `RecedingHorizonNash` and with the Benchmarks `Experiment`.
2. **The solution contract (Base).** Model-based solvers return a [`GNEPSolution`](../basics/solutions.md) (or a problem-specific solution type) with trajectories, strategies, `converged` and `solver_info`. Evaluators and plots read this object only.

The learning route does **not** go through `solve`: `MADDPGSolver` is a plain configuration struct and is run with `train!(solver, env)`, which returns agents and learning histories rather than a `GNEPSolution`. A [guided policy search](../marl/gps.md) step is the point where the two worlds meet: it calls `FNELQ` from the Solvers package on a local LQ approximation of the *same* Base game to produce guidance for the learner.

## Which package do I need?

| I want to… | Use |
|------------|-----|
| formulate a game and solve it with a model-based method | `DifferentialGames` (Base + Solvers) |
| write a new problem type or a new solver in my own package | `DifferentialGamesBase` only |
| train neural-network policies on a differential game | `DifferentialGamesMARL` |
| compare solvers on standard cases, or test robustness to noise/disturbance/partial observation | `DifferentialGamesBenchmarks` |
| recover cost parameters from observed play | `DifferentialGames` ([`InverseLQGames`](../solvers/invlq.md)) |

## A worked route across packages

The example below builds one game in Base, solves it with three different solvers from the Solvers package, and compares the results on the same footing. The learning and benchmarking routes use the identical game object; their code is shown on the [MADDPG](../marl/maddpg.md), [environment](../marl/environments.md) and [Benchmarks](benchmarks.md) pages.

```@example eco
using DifferentialGames, LinearAlgebra

# A linear-quadratic game (Base)
n = 4
game = LQGameProblem(
    Matrix(0.95 * I(n)),
    [Matrix(1.0I(n))[:, 1:1], Matrix(1.0I(n))[:, 3:3]],
    [Matrix(1.0I(n)), Matrix(1.0I(n))],
    [fill(0.1, 1, 1), fill(0.1, 1, 1)],
    [Matrix(10.0I(n)), Matrix(10.0I(n))],
    ones(n), 2.0; dt = 0.1,
)

# Three solvers from the Solvers package, one interface
sols = Dict(name => solve(game, solver) for (name, solver) in
            ("FNELQ" => FNELQ(), "iLQGames" => iLQGames(), "ALGAMES" => ALGAMES()))

for (name, s) in sols
    println(rpad(name, 9), " type = ", rpad(s.equilibrium_type, 14), " converged = ", s.converged,
            "  J = ", round.(get_costs(s); digits = 4))
end
```

`FNELQ` returns the exact feedback Nash costs; `iLQGames` iterates towards the same value with a regularization-limited rate; `ALGAMES` returns an *open-loop* equilibrium, whose cost need not coincide with the feedback one in general. The [conventions page](conventions.md) lists what is and is not comparable across solvers.
