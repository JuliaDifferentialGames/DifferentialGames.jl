# Building Problems

This page describes how to construct game problems from scratch using the player-based API.

## The Player-Based API

The recommended way to build multi-player games is through `PlayerSpec` objects. Each `PlayerSpec` bundles everything about a single player: state/control dimensions, initial state, dynamics function, and objective.

```julia
using DifferentialGames

player = PlayerSpec(
    id,          # unique integer identifier
    n,           # state dimension
    m,           # control dimension
    x0,          # initial state Vector{T}
    dynamics,    # (xi, ui, p, t) -> dxi
    objective,   # PlayerObjective
    constraints  # optional Vector of AbstractPrivateConstraint
)
```

## Defining Dynamics

Dynamics are specified as Julia functions with signature `(xi, ui, p, t) -> dxi`:

```julia
# Double integrator: [x; v] → [v; u]
dyn_1d = (xi, ui, p, t) -> [xi[2]; ui[1]]

# Unicycle: [x, y, v, θ] → [v cos θ, v sin θ, a, ω]
dyn_unicycle = (xi, ui, p, t) -> [xi[3]*cos(xi[4]); xi[3]*sin(xi[4]); ui[1]; ui[2]]

# Continuous-time right-hand sides are discretized by the solver (iLQGames: `ZOHDiscretization()` by default)
```

The function operates on a **single player's** state `xi` and controls `ui`. DifferentialGamesBase assembles the separable joint dynamics automatically. **Cost and constraint functions are solver dependent**: `ALGAMES` passes them the joint state and control, `iLQGames` the player's own slices (see [Conventions](../ecosystem/conventions.md)).

## Defining Costs

### LQ Costs (recommended for linear/near-linear problems)

```julia
# Full Q and R matrices
stage    = LQStageCost(Q, R)           # Q is n×n, R is m×m
terminal = LQTerminalCost(Qf)          # Qf is n×n

# Diagonal shorthand
stage    = DiagonalLQStageCost(q_diag, r_diag)   # weight vectors
terminal = DiagonalLQTerminalCost(qf_diag)
```

### Nonlinear Costs

```julia
stage = NonlinearStageCost(
    (x, u, p, t) -> x' * Q * x + u' * R * u + penalty(x);
    is_separable = true   # true if cost depends only on xi, ui (not other players)
)

terminal = NonlinearTerminalCost((x, p) -> x' * Qf * x)   # terminal costs take (x, p)
```

### Cost Term DSL

For composable nonlinear costs:

`player_slice(x, offset, dim)` extracts a sub-vector of a state vector `x`; the cost terms below take a `state_slice` argument (see the API reference for exact signatures, which were not re-verified in this audit):

```julia
goal     = [0.0, 0.0, 0.0, 0.0]
slice    = 1:4                      # indices of player 1's state in the joint state

stage = minimize(
    track_goal(goal, 1.0; state_slice=slice),      # ‖x - goal‖²
    regularize_input(0.1; control_dim=2),           # ‖u‖²
)

terminal = minimize(
    terminal_goal(goal, 10.0; state_slice=slice)
)
```

### Building PlayerObjective

```julia
obj = PlayerObjective(player_id, stage_cost, terminal_cost)
```

## Defining Constraints

### Private Constraints (per player)

```julia
# Control bounds: -u_max ≤ u ≤ u_max
cb = control_bounds(player_id;
    control_offset = 0,             # offset of this player's controls in joint u
    control_dim    = m,
    lower          = fill(-u_max, m),
    upper          = fill( u_max, m)
)

# State bounds
sb = state_bounds(player_id;
    state_offset = 0,
    state_dim    = n,
    lower        = fill(-x_max, n),
    upper        = fill( x_max, n)
)

# Attach to PlayerSpec
player = PlayerSpec(id, n, m, x0, dyn, obj, [cb, sb])
```

### Shared Constraints (multi-player)

```julia
# Proximity (avoid): ‖pᵢ - pⱼ‖ ≥ d_min
col = collision_avoidance([1, 2];
    i_offset = 0,       # start of player 1's position in the joint state
    j_offset = n,       # start of player 2's position in the joint state
    pos_dim  = 2,       # only position dimensions count
    d_min    = 1.5,
)

# General nonlinear inequality
nl = SharedInequality([1, 2, 3];
    func = (x, u, p, t) -> [constraint_value(x, u)],
    dim  = 1
)
```

## Assembling the Game

### Partially-Decoupled GNEP (most common)

```julia
# With shared constraints
game = PDGNEProblem([player1, player2, player3], [collision12, collision13], tf, dt)

# Unconstrained
game = PDGNEProblem([player1, player2], tf, dt)
```

### Shared-State LQ Game

```julia
game = LQGameProblem(A, B, Q, R, Qf, x0, tf; dt=dt)
```

`A` must be a `Matrix{Float64}` (not `I(n)`); `B`, `Q`, `R`, `Qf` are `Vector{Matrix{Float64}}` — one matrix per player. `x0` is a `Vector{Float64}` and `tf` a `Float64`.

### Inspecting the Game

```julia
num_players(game)            # → 2
state_dim(game)              # → total joint state dim
state_dim(game, 1)           # → state dim of player 1
control_dim(game)            # → total joint control dim
n_steps(game)                # → number of time steps
is_lq_game(game)             # → Bool
is_pd_gnep(game)             # → Bool
has_shared_constraints(game) # → Bool
is_unconstrained(game)       # → Bool
```

## Using remake

`remake` creates a modified copy of a problem with selected fields changed:

```julia
game2 = remake(game;
    initial_state = new_x0,
    time_horizon  = DiscreteTime(5.0, 0.1)
)
```

This is useful in optimization loops where the game structure is fixed but initial conditions vary.
