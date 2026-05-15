# Constraints API

## Shared Constraints

Shared (inter-player) constraints couple multiple players and are enforced jointly.

```@docs
SharedInequality
SharedEquality
```

### Writing Shared Constraint Functions

Shared constraint functions receive the **joint** state vector and joint control vector:

```julia
# Collision avoidance: ‖x₁[1:2] - x₂[1:2]‖² ≥ d²
# Joint state: x = [x₁; x₂] with x₁ ∈ ℝ⁴, x₂ ∈ ℝ⁴
col = SharedInequality([1, 2];
    func = (x, u, p, t) -> [0.5^2 - sum((x[1:2] - x[5:6]).^2)],
    dim  = 1
)
```

The `func` should return a vector of length `dim` where each element is ``\leq 0`` when the constraint is satisfied (for inequality constraints). Use `game.metadata.state_offsets` to index into the joint state reliably:

```julia
offset = game.metadata.state_offsets[2]   # start of player 2's state
col = SharedInequality([1, 2];
    func = (x, u, p, t) -> [d² - sum((x[1:2] - x[(offset+1):(offset+2)]).^2)],
    dim  = 1
)
```

## Private Constraints

Private constraints apply to a single player's state or controls and are propagated automatically to that player's subproblem.

```@docs
ControlBounds
control_bounds
```

### Example: Control Box Constraints

```julia
# Player 1 has 2 controls bounded in [-1, 1]
bounds = control_bounds(1;
    control_offset = 0,
    control_dim    = 2,
    lower          = [-1.0, -1.0],
    upper          = [1.0,  1.0]
)

player1 = PlayerSpec(1, 4, 2, x0, dyn, obj, [bounds])
```
