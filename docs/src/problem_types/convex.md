# ConvexGameProblem

A **convex game** has objectives that are convex in each player's own decision and closed convex constraint sets. Existence of a Nash equilibrium follows from Rosen's theorem; uniqueness holds under *diagonal strict convexity* of the pseudo-gradient ([Rosen1965](@cite)). With a shared coupling constraint the natural solution concept is a **variational generalized Nash equilibrium** ([FacchineiKanzow2010](@cite)).

## Constructors

```@docs
ConvexGameProblem
```

Convexity is verified automatically for LQ objectives and for constraints tagged `is_convex` (box bounds). For nonlinear objectives or untagged constraints pass `assume_convex = true`, which asserts convexity without checking it.

## Example: a static two-player game

[`YiPavel`](../solvers/yipavel.md) works on *static* decisions, so a game with one time step is used and each player's control is its decision. The costs are supplied to `solve` through `cost_fns`, one function of all players' decisions per player:

```math
f_1(u_1,u_2)=\tfrac12(u_1-2)^2+\tfrac c2(u_1+u_2)^2,\qquad
f_2(u_1,u_2)=\tfrac12(u_2-1)^2+\tfrac c2(u_1+u_2)^2,\quad c=0.1 .
```

Setting both gradients to zero gives the Nash equilibrium ``(u_1^*,u_2^*)=(1.75,\,0.75)``.

```@example convex
using DifferentialGames, LinearAlgebra

T = Float64
frozen = (x, u, p, t) -> zeros(T, length(x))                # trivial dynamics: the decision is the control
mkplayer(i) = PlayerSpec{T}(i, 1, 1, zeros(T, 1), frozen,
    PlayerObjective(i, LQStageCost(zeros(1, 1), fill(1e-8, 1, 1)), LQTerminalCost(zeros(1, 1))), T[])

game = ConvexGameProblem([mkplayer(1), mkplayer(2)], T(1.2), T(1.0); assume_convex = true)   # tf, dt: one step

c  = 0.1
f1 = (s1, s2) -> 0.5 * (s1[1] - 2.0)^2 + 0.5 * c * (s1[1] + s2[1])^2
f2 = (s1, s2) -> 0.5 * (s2[1] - 1.0)^2 + 0.5 * c * (s1[1] + s2[1])^2

sol = solve(game, YiPavel(; max_iter = 5000, τ = 0.05, ν = 0.02, σ = 0.05); cost_fns = Function[f1, f2])
u = [get_trajectory(sol, i).controls[1] for i in 1:2]
@assert isapprox(u, [1.75, 0.75]; atol = 1e-2)
(converged = sol.converged, iterations = sol.iterations, decisions = round.(u; digits = 4))
```

The solution is compared with the analytical equilibrium, which is the right check for a method whose stopping rule is a step size.

## Related

- [YiPavel](../solvers/yipavel.md): the distributed primal-dual algorithm for this problem class.
- [Equilibrium concepts](../math/equilibrium_concepts.md): generalized, variational and normalized equilibria.
