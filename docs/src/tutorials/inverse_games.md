# Inverse Games

**Inverse game theory** asks: given observations of agents behaving according to a Nash equilibrium, what are their cost functions?

This is the dual of the forward problem. Instead of computing a Nash equilibrium given known objectives, we infer unknown objectives from trajectory data. Applications include:

- **Intent prediction** for autonomous vehicles
- **Social force estimation** in pedestrian models
- **Preference learning** from expert demonstrations

## Framework

DifferentialGamesBase provides the infrastructure for inverse games. Concrete inverse solvers (e.g., an EnKF-based MONGOOSE solver) are implemented separately and plug into this interface.

An `InverseGameProblem` encodes:

- The dynamics and structure of the game (same as a forward problem)
- Which players have **known** objectives and which are **unknown**
- An **observation model** mapping joint states to measurements
- A **forward solver wrapper** used to simulate hypothesized equilibria

## Example: Recovering a Cost Weight from a Trajectory

We set up a scenario where player 2's cost weight is unknown and must be recovered from observations of a joint trajectory.

```julia
using DifferentialGames, LinearAlgebra

# ── Define the game structure ──────────────────────────────────
dyn = (xi, ui, p, t) -> [xi[2]; ui[1]]   # double integrator

# Player 1: known cost (ego agent)
stage1    = DiagonalLQStageCost([1.0, 0.1], [0.01])
terminal1 = DiagonalLQTerminalCost([10.0, 1.0])
obj1      = PlayerObjective(1, stage1, terminal1)

player1 = PlayerSpec(1, 2, 1, [1.0, 0.0], dyn, obj1)

# Player 2: unknown cost (the agent we're observing)
# We know the form but not the weights — the solver must infer them.
player2 = PlayerSpec(2, 2, 1, [-1.0, 0.0], dyn,
                     PlayerObjective(2,    # placeholder — will be replaced during inference
                         DiagonalLQStageCost([1.0, 0.1], [0.01]),
                         DiagonalLQTerminalCost([10.0, 1.0])))

# ── Knowledge encoding ────────────────────────────────────────
knowledge = [
    KnownObjective(obj1),     # player 1's objective is fixed
    UnknownObjective(),        # player 2's objective must be inferred
]

# ── Observation model ─────────────────────────────────────────
# Full-state observation (noiseless): observe the joint state directly
obs_model = FullStateObservation(4)   # 4 = n1 + n2 = 2 + 2

# Alternatively, noisy partial observation:
# obs_model = NoisyObservation(x -> x[1:2], 1e-3 * I(2), 2)

# ── Forward solver wrapper ────────────────────────────────────
# The inverse solver calls predict_next_state inside its inference loop.
# Implement this using any forward solver.

struct FNELQWrapper <: ForwardSolverWrapper
    # cache warmstart data here if needed
end

function DifferentialGames.predict_next_state(
    ::FNELQWrapper,
    prob::GameProblem{T},
    x0::AbstractVector{T}
) where {T}
    sol = solve(prob, FNELQ())
    return first_step_state(sol)
end

function DifferentialGames.solve_forward(
    ::FNELQWrapper,
    prob::GameProblem{T},
    x0::AbstractVector{T}
) where {T}
    return solve(prob, FNELQ())
end

# ── Inverse problem ───────────────────────────────────────────
inv_prob = InversePDGNEProblem(
    [player1, player2],
    knowledge,
    obs_model,
    FNELQWrapper(),
    3.0,   # tf
    0.1    # dt
)

println("Unknown players: ", unknown_players(inv_prob))     # [2]
println("Known players:   ", known_players(inv_prob))       # [1]

# ── Check as_forward_problem ──────────────────────────────────
# Substitute a hypothesized objective for player 2 and solve forward
hyp_obj2 = PlayerObjective(2,
    DiagonalLQStageCost([2.0, 0.5], [0.05]),   # a different weight hypothesis
    DiagonalLQTerminalCost([5.0, 1.0]))

fwd = as_forward_problem(inv_prob, Dict(2 => hyp_obj2))
fwd_sol = solve(fwd, FNELQ())
println("Forward solution under hypothesis: J₂ = ", get_cost(fwd_sol, 2))
```

## Observation Data

Use `ObservationData` to accumulate observed states over time:

```julia
obs_data = ObservationData{Float64}()

# Simulate or record observations
for t in 0.0:0.1:2.9
    x_observed = randn(4)   # replace with real measurements
    push_observation!(obs_data, x_observed, t)
end

println("Observations collected: ", length(obs_data))

# Access stored data
x_at_t0 = obs_data.states[1]
t_at_0  = obs_data.times[1]
```

## Implementing an Inverse Solver

To implement an inverse solver, use `InverseSolverState` to hold mutable state and return an `InverseGameSolution`:

```julia
# Skeleton of a batch inverse solver (e.g., STLS or gradient-based)
function my_inverse_solve(
    prob::InverseGameProblem{T},
    obs_data::ObservationData{T};
    n_weights::Int = 3
) where {T}
    state  = InverseSolverState{T}()
    t_start = time()

    # ... inference loop ...
    recovered_weights = Dict{Int, Vector{T}}(2 => zeros(T, n_weights))

    return InverseGameSolution(
        prob,
        recovered_weights,                                 # final weights
        Dict{Int, Matrix{T}}(2 => zeros(T, n_weights, 0)), # weight history
        Dict{Int, Array{T,3}}(),                           # ensemble history
        nothing,                                           # forward solution
        true,                                              # converged
        time() - t_start,
        Dict{Symbol, Any}()
    )
end
```

## Observation Models

| Model | Constructor | Notes |
|-------|-------------|-------|
| Full state (noiseless) | `FullStateObservation(n_total)` | Offline / batch settings |
| Additive Gaussian noise | `NoisyObservation(h, R, obs_dim)` | Online / sequential inference |

```julia
# Range + bearing measurement (4D output)
function h_range_bearing(x)
    δx, δy = x[1] - x[3], x[2] - x[4]   # relative position
    r = sqrt(δx^2 + δy^2) + 1e-8
    return [r; δx/r; δy/r; atan(δy, δx)]
end
R_meas = Diagonal(1e-4 * ones(4))
obs = NoisyObservation(h_range_bearing, Matrix(R_meas), 4)
```

## Further Reading

- [`InverseGameProblem` reference](../problem_types/inverse_game.md) — full API for inverse problem types
- [Inverse Games API reference](../api/inverse_games.md) — complete docstring reference
