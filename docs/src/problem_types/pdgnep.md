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

## Example: Two Vehicles with Collision Avoidance

Shared constraints require a constraint-capable solver: **ALGAMES**. (`iLQGames` asserts `iLQGames handles only unconstrained games`.)

```@example pdgnep
using DifferentialGames, LinearAlgebra

dyn = (xi, ui, p, t) -> ui                       # single integrator

make_player(i, x0, goal) = PlayerSpec(i, 2, 2, x0, dyn,
    PlayerObjective(i,
        NonlinearStageCost((x, u, p, t) -> 0.01 * sum(abs2, u[2(i-1)+1:2i]); is_separable = true),   # own controls
        # ALGAMES passes the JOINT state and control to cost functions; player i's blocks start at 2(i-1)
        NonlinearTerminalCost((x, p) -> 10.0 * ((x[2(i-1)+1] - goal[1])^2 +
                                                (x[2(i-1)+2] - goal[2])^2))))

players = [make_player(1, [0.0,  2.0], [5.0, -2.0]),
           make_player(2, [0.0, -1.5], [5.0,  2.0])]

# Collision avoidance: ‖p₁ - p₂‖ ≥ 1.0   ⇔   1.0² - ‖p₁ - p₂‖² ≤ 0
col = SharedInequality([1, 2];
    func = (x, u, p, t) -> [1.0^2 - sum(abs2, x[1:2] - x[3:4])],
    dim  = 1)

game = PDGNEProblem(players, [col], 4.0, 0.1)
sol  = solve(game, ALGAMES(; tol_opt = 1e-2))
sol.converged
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

When writing shared constraint **or cost** functions, use these offsets to index into the joint state:

```julia
# Player 2's position is at offset n₁ in the joint state
offset2 = game.metadata.state_offsets[2]
pos2    = x[(offset2+1):(offset2+2)]
```

!!! warning "Which state does a cost function receive?"
    It depends on the solver. `ALGAMES` passes the **joint** state and control to `NonlinearStageCost`/`NonlinearTerminalCost` functions and to shared constraints, so a terminal cost for player 2 written with `x[1]`, `x[2]` penalizes *player 1's* position. `iLQGames` (through `expand`) passes only **player i's own** state and control slices, so the same function raises a `BoundsError` (or silently differs). Write the function for the solver you use, or use the portable idiom in [Conventions](../ecosystem/conventions.md).

## Private Constraints

Private constraints are attached per-player and propagated automatically:

```julia
bounds = control_bounds(1; control_offset=0, control_dim=2,
                         lower=[-1.0, -1.0], upper=[1.0, 1.0])   # offset = start of THIS player's controls in the joint control vector
player1 = PlayerSpec(1, 4, 2, x0, dyn, obj, [bounds])
game    = PDGNEProblem([player1, player2], 4.0, 0.1)

# Accessing all private constraints:
game.private_constraints   # flat Vector of AbstractPrivateConstraint
```

## Compatible Solvers

Driven by each solver's `solver_capabilities` (see [Solver Selection](../basics/solver_selection.md)):

| Solver | Shared/private constraints | Notes |
|--------|-----------------------------|-------|
| [`iLQGames`](../solvers/ilqgames.md) | **No** (asserts) | Unconstrained nonlinear games. |
| [`ALGAMES`](../solvers/algames.md) | Yes | Open-loop equilibrium approximation; the only built-in solver for constrained `PDGNEProblem`s. |
| [`FNELQ`](../solvers/fnelq.md) | No | **Not** applicable: `PDGNEProblem` uses function dynamics (`SeparableDynamics`), and `FNELQ` requires `LinearDynamics`. Use `LQGameProblem`. |
| [`YiPavel`](../solvers/yipavel.md), [`LIBR`](../solvers/libr.md) | Method-specific | Require `ConvexGameProblem` / `LexicographicGameProblem` respectively. |
