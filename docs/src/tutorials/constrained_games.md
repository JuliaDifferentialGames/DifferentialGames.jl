# Constrained Games

Most practical multi-agent problems have hard constraints: control limits, safety corridors, collision avoidance. DifferentialGames.jl supports two kinds of constraints:

- **Private constraints** — affect only one player (control bounds, state limits)
- **Shared constraints** — couple multiple players (collision avoidance, formation keeping)

The **ALGAMES** solver handles both via an augmented Lagrangian outer loop with Newton-type inner iterations. The **iLQGames** solver does *not*: it asserts on games with constraints.

## Constraint Types

| Type | Constructor | Applies to |
|------|-------------|------------|
| Control box bounds | `control_bounds` | Single player |
| State box bounds | `state_bounds` | Single player |
| Pairwise distance | `collision_avoidance`, `keep_in_range` | Two players |
| Linear coupling | `linear_coupling` | Multiple players |
| General nonlinear inequality | `SharedInequality`, `PrivateInequality` | Shared / private |
| General nonlinear equality | `SharedEquality` | Multiple players |

## Example: Two Vehicles Swapping Lanes

Two single-integrator vehicles drive from ``x=0`` to ``x=5`` and must swap lanes (``y=2 \to -2`` and ``y=-1.5 \to 2``) while staying at least 1.5 apart. Velocities are bounded by ``|u| \le 2``. The problem is asymmetric on purpose (the two starting lanes are not mirror images): a perfectly symmetric swap makes the equilibrium degenerate.

!!! warning "ALGAMES evaluates costs and constraints on the *joint* state and control"
    The joint state is ``x = [x_1; x_2]``, so player ``i``'s position lives at indices `2(i-1)+1 : 2(i-1)+2`, and its controls at `u[2(i-1)+1 : 2i]`. A cost or constraint written with `x[1:2]` for every player refers to player 1, and `sum(abs2, u)` charges a player for *all* players' control effort. (`iLQGames`, which does not support constraints, passes a player's own state and control slices instead; see [Conventions](../ecosystem/conventions.md).)

```@example constrained
using DifferentialGames, LinearAlgebra

dyn = (xi, ui, p, t) -> ui                       # single integrator: ẋ = u

# Player i penalizes only its own block of the joint state.
function make_player(i, x0, goal)
    o = 2 * (i - 1)
    stage    = NonlinearStageCost((x, u, p, t) -> 0.01 * sum(abs2, u[o+1:o+2]); is_separable = true)   # own controls only
    terminal = NonlinearTerminalCost((x, p) -> 10.0 * ((x[o+1] - goal[1])^2 + (x[o+2] - goal[2])^2))
    bounds   = control_bounds(i; control_offset = o, control_dim = 2,
                              lower = [-2.0, -2.0], upper = [2.0, 2.0])
    return PlayerSpec(i, 2, 2, x0, dyn, PlayerObjective(i, stage, terminal), [bounds])
end

players = [make_player(1, [0.0,  2.0], [5.0, -2.0]),
           make_player(2, [0.0, -1.5], [5.0,  2.0])]

# Shared constraint g(x) ≤ 0  ⇔  ‖p₁ − p₂‖ ≥ 1.5
d_min = 1.5
separation = SharedInequality([1, 2];
    func = (x, u, p, t) -> [d_min^2 - sum(abs2, x[1:2] - x[3:4])],
    dim  = 1)

game = PDGNEProblem(players, [separation], 4.0, 0.1)
sol  = solve(game, ALGAMES(; tol_opt = 1e-2))
(converged = sol.converged, iterations = sol.iterations)
```

!!! note "Choosing `tol_opt`"
    With the default `tol_opt = 1e-4` this problem reaches a trajectory that satisfies the constraints to ~1e-7 but the stationarity residual stays large and `converged` is `false`. Tolerances should be chosen to match the accuracy you need; always verify the constraints directly, as below, rather than trusting a flag.

## Checking Constraint Satisfaction

Solution `states` are joint, so the full joint state at **every time node** can be passed directly to `evaluate_constraint`. By default ALGAMES enforces constraints only at nodes `1:N` (see `terminal_constraints` below), so checking all `N+1` nodes is the stricter test:

```@example constrained
X = get_trajectory(sol, 1).states                      # joint state, 4 × (N+1)
U = hcat(get_trajectory(sol, 1).controls, get_trajectory(sol, 2).controls)

# max over all nodes of g(x_k) (≤ 0 means feasible)
g_max = maximum(evaluate_constraint(separation, X[:, k], zeros(4), nothing, 0.0)[1]
                for k in axes(X, 2))

# minimum inter-vehicle distance over the horizon
d_obs = minimum(norm(X[1:2, k] - X[3:4, k]) for k in axes(X, 2))

@show g_max d_obs maximum(abs, U)
@assert g_max ≤ 1e-3                  # constraint satisfied to tolerance
@assert maximum(abs, U) ≤ 2.0 + 1e-2  # control bounds respected
```

Here `g_max < 0` and the minimum distance is well above `d_min`: **the collision constraint is inactive in this particular solution** (the vehicles cross the lane centre at different times). The control bound, in contrast, is active (`max|u| ≈ 2`). A different local solution (for example the one found with the default `tol_opt`) can pass closer to the constraint boundary, which is why the check above is worth running.

## Private Constraints in Detail

Private constraints are attached to a `PlayerSpec` and act on that player's sub-state and controls:

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

ALGAMES options (see the [ALGAMES reference](../solvers/algames.md)):

```julia
sol = solve(game, ALGAMES(;
    outer_iter = 50,     # maximum augmented-Lagrangian iterations
    inner_iter = 10,     # Newton steps per outer iteration
    ρ_init     = 1.0,    # initial penalty
    ρ_increase = 10.0,   # penalty growth factor
    ρ_max      = 1e8,    # penalty cap
    reg        = 1e-3,   # Newton regularization
    tol_opt    = 1e-4,   # stationarity tolerance
    tol_dyn    = 1e-4,   # dynamics-residual tolerance
    tol_con    = 1e-3,   # constraint-violation tolerance
))
```

`verbose` is an argument of `solve`, not of `ALGAMES`: `solve(game, ALGAMES(); verbose=true)`.

### Constraints at the final time

By default ALGAMES does not constrain the terminal state `x(N+1)`. Pass `terminal_constraints = true` to enforce the state-only constraint rows there as well (control bounds, which depend on `u`, still apply to nodes `1:N` only):

```julia
sol = solve(game, ALGAMES(; tol_opt = 1e-2, terminal_constraints = true))
```

If ALGAMES fails to converge: check that the initial state is feasible, loosen `tol_opt`, try a different `ρ_increase`/`inner_iter`, and **always** verify the result independently. Different settings can land in different local solutions of the same problem.

## Further Reading

- [ALGAMES solver reference](../solvers/algames.md) — option list and algorithm details
- [Constraint API reference](../api/constraints.md) — all constraint types and constructors
