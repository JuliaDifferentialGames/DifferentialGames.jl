# LexicographicGameProblem

A **lexicographic general-sum game** gives each agent a two-part cost, ordered lexicographically: first a *collision* cost, then a *personal* cost (reaching a goal, saving fuel):

```math
J_i(z) = \big(J_i^{\mathrm{col}}(z),\; J_i^{\mathrm{per}}(z)\big),\qquad
J_i^{\mathrm{col}} = \sum_{j\ne i} f_{ij}(z_i,z_j),\quad
J_i^{\mathrm{per}} = g_i(z_i) - \sum_{j\ne i} g_j(z_j),
```

with ``f_{ij}=f_{ji}\ge 0``. A lexicographic game is an *ordinal potential game*, which guarantees a pure-strategy Nash equilibrium ([MillerMitra2022](@cite)). The type wraps a `PDGNEProblem` (`forward_game`) that supplies dynamics, initial states and horizon.

## Constructor

```@docs
LexicographicGameProblem
```

## Example: two agents swapping sides

The cost functions take a player's trajectory `z_i = (X_i, U_i)` with `X_i` the private state (`nᵢ × (N+1)`) and `U_i` the controls (`mᵢ × N`).

```@example lex
using DifferentialGames, LinearAlgebra

T   = Float64
dyn = (x, u, p, t) -> [x[3], x[4], u[1], u[2]]               # double integrator, [px, py, vx, vy]
placeholder = PlayerObjective(1, LQStageCost(zeros(4, 4), Matrix{T}(I, 2, 2) * 1e-6),
                                 LQTerminalCost(zeros(4, 4)))   # LIBR ignores the LQ costs
mkplayer(id, x0) = PlayerSpec{T}(id, 4, 2, x0,  dyn,
                                 PlayerObjective(id, placeholder.stage_cost, placeholder.terminal_cost), T[])
p1 = mkplayer(1, [ 1.0,  0.05, 0.0, 0.0])      # the small lateral offset breaks the head-on symmetry
p2 = mkplayer(2, [-1.0, -0.05, 0.0, 0.0])

# Pairwise collision cost f_ij ≥ 0: squared violation of a 0.3 separation, summed over the trajectory
d2 = 0.3^2
f_col = (zi, zj) -> begin
    Xi, _ = zi; Xj, _ = zj
    sum(max(zero(eltype(Xi)), d2 - (Xi[1, k] - Xj[1, k])^2 - (Xi[2, k] - Xj[2, k])^2)^2
        for k in 1:size(Xi, 2))
end
no_cost = (zi, zj) -> zero(eltype(zi[1]))                     # unused diagonal
collision_pairs = Function[no_cost f_col; f_col no_cost]

# Personal costs g_i: reach the goal with little effort
goal_cost(goal) = zi -> begin
    Xi, Ui = zi
    sum((Xi[1, k] - goal[1])^2 + (Xi[2, k] - goal[2])^2 for k in 1:size(Xi, 2)) + 0.1 * sum(abs2, Ui)
end

game = LexicographicGameProblem([p1, p2], collision_pairs,
                                Function[goal_cost([-1.0, 0.0]), goal_cost([1.0, 0.0])], 1.0, 0.1)
game
```

Solve it with [`LIBR`](../solvers/libr.md):

```@example lex
# The priority of the collision cost is enforced through a quadratic penalty: use a large weight.
sol = solve(game, LIBR(; max_iter = 100, inner_iter = 200, step_size = 0.01, col_penalty = 1e9))

# LIBR stores each player's OWN state in its trajectory (state_trajectory is `nothing`)
x1 = get_trajectory(sol, 1).states
x2 = get_trajectory(sol, 2).states
separation = minimum(norm(x1[1:2, k] - x2[1:2, k]) for k in axes(x1, 2))
@assert separation > 0.28          # the first-priority (collision) cost is respected
(converged = sol.converged, iterations = sol.iterations, min_separation = separation,
 collision_cost = sol.solver_info[:J_col])
```

Check the *meaning* of the result, not just the flag: the minimum separation is close to the 0.3 collision distance (the collision cost is the first priority) while the agents still move towards their goals on opposite sides. The stopping test (`converged`) is on the change of the control trajectories.

!!! warning "The priority is enforced by a penalty"
    With the default `col_penalty = 1e3` the same game gives a minimum separation of about 0.17: the penalty is too weak to keep the collision cost lexicographically first, and the solver trades it against the personal cost. The value `1e9` above restores the ordering (separation ≈ 0.29, collision cost ≈ 2·10⁻⁵). Always check the first-priority cost of the returned solution.

## Accessors

`collision_cost(game, i, z_i, z_minus_i)`, `personal_cost(...)`, `lexicographic_cost(...)` (the pair) and `ordinal_potential(game, z_by_player)` evaluate the cost components.
