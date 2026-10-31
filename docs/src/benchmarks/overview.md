# Benchmark Suite

The ecosystem ships a set of benchmark problems so that solvers, learned policies and extensions can be compared on the same games. This page is the catalogue; each family has its own page.

!!! note "Where the problems live"
    The cases that need a *scoring* harness (versioned identity, independent metrics, challenge transforms, plotting) are in [`DifferentialGamesBenchmarks.jl`](../ecosystem/benchmarks.md). The multi-agent particle tasks of the MADDPG paper are in [`DifferentialGamesMARL.jl`](particle_tasks.md) because they are built for learning, with observation functions and reset distributions; they are still ordinary Base games.

## Catalogue

| Family | Problem | Where | Players | Dynamics | Stresses |
|--------|---------|-------|---------|----------|----------|
| Classical games | Simple pursuit | Benchmarks, `SimplePursuit.Nominal_v1` | 2 | single integrators | minimal pursuit–evasion |
|  | Homicidal chauffeur | `HomicidalChauffeur.Nominal_v1` | 2 | bounded-turn car vs free point | dynamics asymmetry |
|  | Game of two cars | `GameOfTwoCars.Nominal_v1` | 2 | two bounded-turn cars | dynamics symmetry |
|  | Lady in the lake | `LadyInTheLake.Nominal_v1` | 2 | free swimmer vs shore-bound pursuer | asymmetric domain |
| Reference | LQ formation | `LQFormation.Nominal_v1` | 2 | linear | exact (Riccati) solution as ground truth |
| FALCON suite | [Hallway passing](hallway.md) | `HallwayPassing.Nominal_v1` | 3 | unicycles (SE(2)) | narrow, non-convex passages; negotiation |
|  | [Hallway with communication range](hallway.md) | `HallwayPassing.Communication_v1` | 3 | unicycles (SE(2)) | as above plus a hard convoy constraint (30 s) |
|  | [Racing](racing.md) | `Racing.AustinNominal_v1` | 2 | dynamic bicycle, Pacejka tyres | stiff nonlinear dynamics, competitive cost |
|  | [Racing, first hairpin](racing.md#The-hairpin-and-the-50-initial-conditions) | `Racing.AustinHairpin_v1` | 2 | dynamic bicycle | the sharp-corner example of the FALCON figure (30 m segment, 4 s) |
| MADDPG tasks | [Cooperative navigation](particle_tasks.md#cooperative_navigation) | MARL | `N` | point masses | cooperation, coverage |
|  | [Predator–prey](particle_tasks.md#predator_prey) | MARL | `N+1` | point masses | competition, obstacles |
|  | [Keep-away](particle_tasks.md#keep_away) | MARL | `N+M` | point masses | information asymmetry (in the observation) |
|  | [Physical deception](particle_tasks.md#physical_deception) | MARL | `N+1` | point masses | deception (in the observation) |

Cooperative communication and covert communication from the same paper are **not** provided; the [particle tasks page](particle_tasks.md#Tasks-that-are-not-implemented) says why.

## How the FALCON problems were ported

The hallway and racing problems come from the validation suite of FALCON (Outland and Arya, *A Fast, Convergent Algorithm for Solving Non-convex Partially-Decoupled Generalized Nash Equilibrium Problems*). FALCON is an external, unregistered solver; the benchmarks here are **solver-agnostic Base games**, so a few modelling choices differ from the FALCON scripts (continuous-time dynamics that each solver discretizes itself, a smooth terminal cost instead of a hard goal ball, no input-rate penalty). Every difference is listed at the top of the corresponding source file and on the problem's page. The geometry, vehicle models and constants are the same.

## What is verified

Verified means run against the current sources.

| Problem | Verified |
|---------|----------|
| Hallway passing | geometry and single-file condition; a reference plan that follows the centre-line through both chokes and reaches the goals; `ALGAMES` run (see the [page](hallway.md) for what it achieves) |
| Racing | spline track (interpolation, C¹, arc length); bicycle dynamics sanity; side-by-side start; a closed-loop reference plan that stays on track; `ALGAMES` run (see the [page](racing.md)) |
| MADDPG tasks | construction, environment stepping, cost definitions, information asymmetry in the observation; `ALGAMES` run on the model-based version |

All of the above are covered by tests in the respective package. What is **not** established: that any solver in the ecosystem converges to a Nash equilibrium of the hallway or racing games. These are hard, non-convex benchmarks, which is why they are in the suite.

## Scoring a solution

[`evaluate`](../ecosystem/benchmarks.md) recomputes metrics from the returned trajectories; a solver's own convergence flag is not trusted. The FALCON problems add metrics that matter for them:

| Problem | Extra metrics |
|---------|---------------|
| Hallway | `:goal_error` (largest final distance from a goal), `:min_separation` (smallest pairwise distance), `:wall_clearance` (smallest signed distance to a wall) |
| Racing | `:progress` (final arc length of each car), `:min_separation`, `:track_clearance`, `:lag` (distance between a car and the centre-line point at its own progress) |

!!! warning "The default `:nash_residual` is not a Nash residual"
    `game_quality_nash_residual` recomputes each player's cost from the trajectory and compares it with the cost the solver *reported*. It detects misreported costs; it does **not** test optimality, and returns `0` for a solution that is far from any equilibrium (observed on the racing game, where the solver's own optimality residual was of order 10³). Treat it as a cost-consistency check. A genuine equilibrium test needs a best response against the other players' fixed strategies, which the benchmark package does not provide yet.

## Adding a problem

A case is a zero-argument factory returning a Base game plus an evaluator and, optionally, a visualizer:

```julia
const MY_CASE_V1 = BenchmarkCase{Float64}(
    CaseID(:MyProblem, :Nominal, v"1"),
    () -> my_game(),                       # builds a fresh Base game every call
    FullModelAccess(),
    "One-sentence description.",
    ["Reference."],
)
register_case!(MY_CASE_V1)
register_visualizer!(MY_CASE_V1.id, my_visual_model)   # optional
```

Follow the existing cases: put the model, the cost and the constraints in `src/cases/`, state at the top of the file how it differs from the source, write cost functions that select the player's own block when handed the joint vector (see [Conventions](../ecosystem/conventions.md)), and test the definition (geometry, dynamics, a reference plan) rather than a solver outcome. See also [Extending the ecosystem](../api/solver_interface.md).
