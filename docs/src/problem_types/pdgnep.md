# PDGNEProblem

A **Partially-Decoupled Generalized Nash Equilibrium Problem** (PD-GNEP). Each player has its own independent state and dynamics; players interact only through costs and shared constraints. This is the standard formulation for multi-robot and multi-vehicle scenarios.

## Constructors

```@docs
PDGNEProblem
```

## With Shared Constraints

```julia
game = PDGNEProblem(players, shared_constraints, tf, dt)
```

| Argument | Type | Description |
|----------|------|-------------|
| `players` | `Vector{PlayerSpec{T}}` | One entry per player |
| `shared_constraints` | `Vector{<:AbstractSharedConstraint}` | Inter-player constraints |
| `tf` | `T` | Final time |
| `dt` | `T` | Time step |

## Without Shared Constraints

```julia
game = PDGNEProblem(players, tf, dt)
```

## Example: Two Spacecraft

```julia
using DifferentialGames, LinearAlgebra

# Double-integrator dynamics (position + velocity)
dyn = (xi, ui, p, t) -> [xi[3]; xi[4]; ui[1]; ui[2]]

make_player(id, x0) = PlayerSpec(
    id, 4, 2, x0, dyn,
    PlayerObjective(id,
        DiagonalLQStageCost([0.0, 0.0, 1.0, 1.0], [0.1, 0.1]),
        DiagonalLQTerminalCost([10.0, 10.0, 1.0, 1.0])
    )
)

player1 = make_player(1, [ 3.0, 0.0, 0.0, 0.0])
player2 = make_player(2, [-3.0, 0.0, 0.0, 0.0])

# Collision avoidance: ‖x₁ - x₂‖ ≥ 1.0 m
col = SharedInequality([1, 2];
    func = (x, u, p, t) -> [1.0^2 - sum((x[1:2] - x[5:6]).^2)],
    dim  = 1
)

game = PDGNEProblem([player1, player2], [col], 4.0, 0.1)
sol  = solve(game, iLQGames())
```

## Joint State Layout

In a PD-GNEP, the joint state vector concatenates all player states in order:

```
x = [x₁; x₂; ...; xₙ]
```

The offsets are recorded in `game.metadata`:

```julia
game.metadata.state_dims     # [n₁, n₂, ...]
game.metadata.state_offsets  # [0, n₁, n₁+n₂, ...]
game.metadata.control_dims   # [m₁, m₂, ...]
game.metadata.control_offsets # [0, m₁, m₁+m₂, ...]
```

When writing shared constraint functions, use these offsets to index into the joint state:

```julia
# Player 2's position is at offset n₁ in the joint state
offset2 = game.metadata.state_offsets[2]
pos2    = x[(offset2+1):(offset2+2)]
```

## Private Constraints

Private constraints are attached per-player and propagated automatically:

```julia
bounds = control_bounds(1; control_offset=0, control_dim=2,
                         lower=[-1.0, -1.0], upper=[1.0, 1.0])
player1 = PlayerSpec(1, 4, 2, x0, dyn, obj, [bounds])
game    = PDGNEProblem([player1, player2], 4.0, 0.1)

# Accessing all private constraints:
game.private_constraints   # flat Vector of AbstractPrivateConstraint
```

## Compatible Solvers

| Solver | Constraints | Notes |
|--------|-------------|-------|
| [`iLQGames`](../solvers/ilqgames.md) | No | Fastest for unconstrained nonlinear games |
| [`ALGAMES`](../solvers/algames.md) | Yes | Required for shared/private constraints |
| [`FNELQ`](../solvers/fnelq.md) | No | Only if all costs are LQ |
