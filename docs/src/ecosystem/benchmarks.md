# DifferentialGamesBenchmarks.jl

`DifferentialGamesBenchmarks.jl` provides **benchmark cases**, **assumption-removal challenges**, **independent evaluation metrics**, **experiment orchestration** and **Makie-based visualization** for the ecosystem. A benchmark case always builds a real Base game, never a private copy, so anything that solves a Base game can be benchmarked.

Its role in the ecosystem is evaluation: it consumes games (Base) and solutions produced by *any* solver that follows the solver contract (Solvers, or your own package), and scores them with metrics that do not rely on the solver's own convergence flag.

!!! note "Status"
    The examples below were run against the current sources (Julia 1.12) and are **not executed** during the documentation build. Plotting requires `CairoMakie` or `GLMakie`, which are extensions.

## Concepts

| Concept | Meaning |
|---------|---------|
| `CaseID(name, variant, version)` | identity of a case, e.g. `SimplePursuit.Nominal_v1` |
| `BenchmarkCase` | an immutable, versioned reference to a case: `id`, a zero-argument `build` factory, a default `AccessSpec`, a description and references; `game(case)` rebuilds a **fresh** Base game every call, so cases are reproducible |
| `CASE_REGISTRY`, `register_case!`, `lookup_case` | registry of cases; the official ones are registered at load time |
| `AccessSpec` (`FullModelAccess`, `ApproximateModelAccess`, `SimulatorOnlyAccess`, `ModelFreeAccess`, `MaskedObjectives`, `ObservationOnlyAccess`) | what the solver is allowed to know about the game; the same case can be offered to model-based and learning-based methods |
| `ChallengeProfile` and transforms `add_process_noise`, `add_disturbance_bound`, `restrict_observation`, `restrict_communication` | **remove assumptions** by wrapping the game in the Base [uncertainty types](../problem_types/uncertainty.md) (`StochasticGameProblem`, `RobustGameProblem`, `POGameProblem`, communication limits) |
| `Evaluator`, `evaluate` | metrics recomputed from the trajectories: *validity* (constraint satisfaction), *quality* (e.g. cost-consistency residual, capture radius), *robustness* (dynamics residual), *efficiency* (solve time and iterations) |
| `Experiment`, `Benchmark`, `Budget`, `run`, `ExperimentResult` | orchestration of case × solver × challenge × budget |

## Official cases

| Case | `CaseID` | What it stresses |
|------|----------|------------------|
| Simple pursuit | `SimplePursuit.Nominal_v1` | minimal single-integrator pursuit–evasion |
| Homicidal chauffeur | `HomicidalChauffeur.Nominal_v1` | dynamics asymmetry: bounded-turning-radius pursuer vs a free evader |
| Game of two cars | `GameOfTwoCars.Nominal_v1` | dynamics symmetry: both players are bounded-turning-radius cars |
| Lady in the lake | `LadyInTheLake.Nominal_v1` | asymmetric domain: free-swimming evader vs a pursuer confined to a 1-D shore |
| LQ formation | `LQFormation.Nominal_v1` | leader/follower LQ Nash game with a known closed-form solution (the analytic reference backbone) |
| Hallway passing | `HallwayPassing.Nominal_v1` | three unicycles through an S-shaped double-choke corridor, from the FALCON suite ([page](../benchmarks/hallway.md)) |
| Racing (Austin) | `Racing.AustinNominal_v1` | two dynamic-bicycle cars racing head-to-head, from the FALCON suite ([page](../benchmarks/racing.md)) |

The four pursuit–evasion cases are built on the PD-GNEP path and are solved with `ALGAMES` (the turning-rate bounds of the chauffeur and two-cars cases need constraint support). The LQ case is solved exactly by `FNELQ`. The hallway and racing cases are hard non-convex problems that no solver here converges on; see the [benchmark suite](../benchmarks/overview.md) for the full catalogue, including the multi-agent particle tasks of the MADDPG paper.

## Example

```julia
using DifferentialGamesBenchmarks, DifferentialGamesBase, DifferentialGamesBaseSolvers

case = SIMPLE_PURSUIT_NOMINAL_V1
g    = game(case)                                          # a Base GameProblem
sol  = DifferentialGamesBase.solve(g, ALGAMES())
res  = evaluate(simple_pursuit_evaluator(), g, sol)

res.quality[:time_to_capture]            # 3.3
res.quality[:capture_radius_min]         # 0.42
res.quality[:nash_residual]              # Dict(1 => 0.0, 2 => 0.0): reported vs recomputed cost, see the warning below
res.validity[:constraints_satisfied]
```

Or through the orchestration layer:

```julia
experiment = Experiment(case, ALGAMES(); evaluator = simple_pursuit_evaluator())
result = DifferentialGamesBenchmarks.run(experiment)
```

Exact solutions give an analytic check of the whole stack: for the LQ formation case

```julia
case = LQ_FORMATION_NOMINAL_V1
sol  = DifferentialGamesBase.solve(game(case), FNELQ())
evaluate(lq_formation_evaluator(), game(case), sol).quality[:formation_error_final]   # ≈ 2·10⁻⁶
```

(Call `run` as `DifferentialGamesBenchmarks.run` to avoid ambiguity with `Base.run`.)

!!! warning "`:nash_residual` is a cost-consistency check"
    Despite its name, the default quality metric `game_quality_nash_residual` compares each player's *recomputed* cost with the cost the solver *reported*. It catches misreported objectives but does not test optimality: it returned `0` for a racing solution whose own optimality residual was of order 10³. A genuine equilibrium test (a best response against the other players' fixed trajectories) is not yet in the package.

## Visualization

With `CairoMakie` (static) or `GLMakie` (interactive) loaded: `gameplot(case, sol; preset = :publication)`, `dashboard(result)`, `animate`, `interactive`, `renderframe`. Visual models are registered per case with `register_visualizer!`.

## Relationship to the rest of the ecosystem

- **Base** supplies the games, the uncertainty wrappers used by challenges and the solution types read by evaluators.
- **Solvers** are used through the `DifferentialGamesBenchmarksSolversExt` extension; any solver following the [solver contract](../api/solver_interface.md) can be benchmarked.
- **MARL**: a case's game can be wrapped in a [`MADDPGEnv`](../marl/environments.md) and a trained agent evaluated in the same way; the `ModelFreeAccess`/`SimulatorOnlyAccess` specifications describe such use.
- The package also served to *find* defects in the solvers: while building the LQ formation case an affine-term sign error in `FNELQ` was identified independently (it is now fixed in the sibling working tree, see the [audit](https://github.com/JuliaDifferentialGames/DifferentialGames.jl/blob/main/docs/AUDIT.md)).
