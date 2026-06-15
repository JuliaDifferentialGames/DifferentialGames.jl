# Constrained Games

Most practical multi-agent problems have hard constraints: control limits, safety corridors, collision avoidance. DifferentialGames.jl supports two kinds of constraints:

- **Private constraints** — affect only one player (control bounds, state limits)
- **Shared constraints** — couple multiple players (collision avoidance, formation keeping)

The **ALGAMES** solver handles both via an augmented Lagrangian outer loop with Newton-type inner iterations.

## Constraint Types

| Type | Constructor | Applies to |
|------|-------------|------------|
| Control box bounds | `control_bounds` | Single player |
| State box bounds | `state_bounds` | Single player |
| Proximity (stay close) | `keep_in_range` | Two players |
| Proximity (avoid) | `collision_avoidance` | Two players |
| Linear coupling | `linear_coupling` | Multiple players |
| General nonlinear inequality | `SharedInequality` | Multiple players |
| General nonlinear equality | `SharedEquality` | Multiple players |

## Example: Formation Flight with Control Limits

Three aircraft maintain a triangular formation while staying within actuator limits.

```julia
using DifferentialGames, LinearAlgebra

# ── Single-integrator aircraft dynamics ───────────────────────
# State: [x, y]  Control: [vx, vy]  (velocity-controlled)
dyn = (xi, ui, p, t) -> ui

# ── Costs ─────────────────────────────────────────────────────
# Each aircraft wants to reach its formation slot
goals = [[4.0, 2.0], [4.0, -2.0], [8.0, 0.0]]   # formation targets

function make_stage(i)
    NonlinearStageCost(
        (x, u, p, t) -> 0.01 * (u' * u),   # minimize fuel only
        is_separable=true
    )
end

function make_terminal(i)
    g = goals[i]
    NonlinearTerminalCost(x -> 10.0 * ((x[1]-g[1])^2 + (x[2]-g[2])^2))
end

# ── Control bounds ────────────────────────────────────────────
# Each player's velocity is limited to ±2 m/s in each axis
function make_bounds(player_id, control_offset)
    control_bounds(player_id;
        control_offset = control_offset,
        control_dim    = 2,
        lower          = [-2.0, -2.0],
        upper          = [ 2.0,  2.0]
    )
end

# ── Pairwise collision avoidance ──────────────────────────────
# Minimum separation: 1.5 m
function avoid_pair(i, j, offset_i, offset_j)
    SharedInequality([i, j];
        func = (x, u, p, t) -> begin
            xi = x[offset_i .+ (1:2)]
            xj = x[offset_j .+ (1:2)]
            [1.5^2 - sum((xi .- xj).^2)]
        end,
        dim = 1
    )
end

# ── Build players ─────────────────────────────────────────────
x0s = [[0.0, 1.0], [0.0, -1.0], [0.0, 0.0]]

players = [
    PlayerSpec(i, 2, 2, x0s[i], dyn,
               PlayerObjective(i, make_stage(i), make_terminal(i)),
               [make_bounds(i, (i-1)*2)])
    for i in 1:3
]

# ── Shared constraints ────────────────────────────────────────
shared = [
    avoid_pair(1, 2, 0, 2),
    avoid_pair(1, 3, 0, 4),
    avoid_pair(2, 3, 2, 4),
]

game = PDGNEProblem(players, shared, 4.0, 0.1)

# ── Solve ─────────────────────────────────────────────────────
sol = solve(game, ALGAMES(; verbose=true))

println("converged: ", sol.converged)
for i in 1:3
    println("Player $i cost: ", round(get_cost(sol, i); digits=4))
    traj = get_trajectory(sol, i)
    println("  final position: ", round.(traj.states[:, end]; digits=2))
end
```

## Checking Constraint Satisfaction

```julia
# Retrieve all shared constraint values at the solution
for (k, c) in enumerate(game.shared_constraints)
    traj1 = get_trajectory(sol, 1)
    traj2 = get_trajectory(sol, 2)
    # Joint state at final time
    x_T = vcat(traj1.states[:, end], traj2.states[:, end])
    val = evaluate_constraint(c, x_T, zeros(4), nothing, 0.0)
    println("Constraint $k at T: ", val, " (should be ≤ 0 for satisfied inequality)")
end
```

## Private Constraints in Detail

Private constraints are attached to a `PlayerSpec` and affect only that player's sub-state and controls:

```julia
# Control bounds for player 1 (control offset 0, 2 controls)
cb = control_bounds(1; control_offset=0, control_dim=2,
                    lower=[-1.0, -1.0], upper=[1.0, 1.0])

# State bounds for player 1 (state offset 0, state dim 4)
sb = state_bounds(1; state_offset=0, state_dim=4,
                  lower=fill(-5.0, 4), upper=fill(5.0, 4))

# General nonlinear private inequality: ‖u‖² ≤ 1
nl = PrivateInequality(1; func=(x,u,p,t) -> [u'*u - 1.0], dim=1)

player = PlayerSpec(1, 4, 2, x0, dyn, obj, [cb, sb, nl])
```

## Solver Tuning

ALGAMES uses an augmented Lagrangian outer loop. Key parameters:

```julia
sol = solve(game, ALGAMES(;
    max_outer   = 30,    # max augmented Lagrangian outer iterations
    max_inner   = 100,   # max Newton iterations per outer loop
    ρ_init      = 1.0,   # initial penalty parameter
    ρ_scale     = 10.0,  # penalty growth factor
    ρ_max       = 1e6,   # penalty cap
    ε_primal    = 1e-4,  # primal feasibility tolerance
    ε_dual      = 1e-4,  # dual feasibility tolerance
    verbose     = false
))
```

If ALGAMES fails to converge, try:
1. Increasing `ρ_scale` for faster constraint enforcement
2. Checking that constraints are feasible from the initial state
3. Reducing the horizon or time step

## Further Reading

- [ALGAMES solver reference](../solvers/algames.md) — full option list and algorithm details
- [Constraint API reference](../api/constraints.md) — all constraint types and constructors
