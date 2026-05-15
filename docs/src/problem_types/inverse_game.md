# InverseGameProblem

An `InverseGameProblem` specifies an inverse game: given observations of agents playing a Nash equilibrium, infer the unknown players' cost functions.

The problem is a pure specification object (immutable, no solver state) following the same design pattern as `GameProblem`.

## Constructors

```@docs
InverseGameProblem
InversePDGNEProblem
```

## Key Types

```@docs
PlayerKnowledge
KnownObjective
UnknownObjective
ObservationModel
FullStateObservation
NoisyObservation
ForwardSolverWrapper
```

## InversePDGNEProblem Constructor

The most common constructor for partially-decoupled inverse games:

```julia
prob = InversePDGNEProblem(
    players,            # Vector{PlayerSpec{T}}
    knowledge,          # Vector{PlayerKnowledge}
    shared_constraints, # AbstractVector (can be empty [])
    observation_model,  # ObservationModel
    forward_solver,     # ForwardSolverWrapper
    tf, dt              # time horizon
)

# Without shared constraints:
prob = InversePDGNEProblem(
    players, knowledge, observation_model, forward_solver, tf, dt
)
```

## Implementing a ForwardSolverWrapper

The forward solver wrapper is called by the inverse solver to predict the next joint state under a hypothesized objective:

```julia
struct MyForwardWrapper <: ForwardSolverWrapper end

function DifferentialGames.predict_next_state(
    ::MyForwardWrapper,
    prob::GameProblem{T},
    x0::AbstractVector{T}
) where {T}
    sol = solve(prob, FNELQ())         # or iLQGames(), etc.
    return first_step_state(sol)       # joint state at k=2
end

function DifferentialGames.solve_forward(
    ::MyForwardWrapper,
    prob::GameProblem{T},
    x0::AbstractVector{T}
) where {T}
    return solve(prob, FNELQ())
end
```

## Accessors

```@docs
unknown_players
known_players
n_unknown
known_objective
as_forward_problem
```

## Example

```julia
using DifferentialGames

dyn = (xi, ui, p, t) -> [xi[2]; ui[1]]

# Known player 1
stage1 = DiagonalLQStageCost([1.0, 0.1], [0.01])
term1  = DiagonalLQTerminalCost([10.0, 1.0])
obj1   = PlayerObjective(1, stage1, term1)

# Player 2: unknown cost
stage2 = DiagonalLQStageCost([1.0, 0.1], [0.01])  # placeholder
term2  = DiagonalLQTerminalCost([10.0, 1.0])
obj2   = PlayerObjective(2, stage2, term2)

p1 = PlayerSpec(1, 2, 1, [1.0, 0.0], dyn, obj1)
p2 = PlayerSpec(2, 2, 1, [-1.0, 0.0], dyn, obj2)

knowledge = [KnownObjective(obj1), UnknownObjective()]
obs       = FullStateObservation(4)

struct FNELQWrapper <: ForwardSolverWrapper end
function DifferentialGames.predict_next_state(::FNELQWrapper, prob, x0)
    return first_step_state(solve(prob, FNELQ()))
end
function DifferentialGames.solve_forward(::FNELQWrapper, prob, x0)
    return solve(prob, FNELQ())
end

inv_prob = InversePDGNEProblem([p1, p2], knowledge, obs, FNELQWrapper(), 3.0, 0.1)

println("Unknown players: ", unknown_players(inv_prob))   # [2]

# Test as_forward_problem with a hypothesis
hyp = Dict(2 => PlayerObjective(2, DiagonalLQStageCost([2.0, 0.2], [0.05]),
                                     DiagonalLQTerminalCost([5.0, 0.5])))
fwd = as_forward_problem(inv_prob, hyp)
sol = solve(fwd, FNELQ())
println("Forward cost under hypothesis: J₂ = ", get_cost(sol, 2))
```

## InverseGameSolution

```@docs
InverseGameSolution
get_weights
get_weight_history
```

## ObservationData

```@docs
ObservationData
push_observation!
```
