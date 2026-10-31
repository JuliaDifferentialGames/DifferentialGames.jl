# LQ Games

Linear-quadratic (LQ) differential games have linear dynamics and quadratic costs. They admit closed-form feedback Nash equilibria via a coupled backward Riccati recursion, making them an important special case: exact, fast, and analytically tractable.

## Background

In a discrete-time, finite-horizon ``P``-player LQ game with shared state ``x_k \in \mathbb{R}^n``, player ``i`` chooses ``u_{i,k}`` to minimize

```math
J_i = \tfrac12\left[\sum_{k=0}^{N-1} \left( x_k^\top Q_i x_k + u_{i,k}^\top R_i u_{i,k} \right) + x_N^\top Q_{f,i} x_N\right]
```

subject to the shared linear dynamics

```math
x_{k+1} = A x_k + \sum_j B_j u_{j,k}.
```

A **feedback Nash equilibrium** is a set of strategies ``u_{i,k} = -K_{i,k} x_k`` such that no player can reduce its cost by unilaterally deviating from its strategy at any stage and state. FNELQ computes the gains ``K_{i,k}`` from the coupled Riccati equations.

!!! note "Cost convention"
    The reported costs include the factor ``\tfrac12`` shown above. The LQR check at the end of this page confirms this numerically.

## Shared-State LQ Game

Use `LQGameProblem` when all players share a single joint state. Matrix arguments may be any `AbstractMatrix` (`I(n)`, `Diagonal`, …); they are converted to dense `Float64` arrays. The examples below use `Matrix(...)` explicitly, which also works with older versions of `DifferentialGamesBase`, whose constructor required concrete `Matrix{Float64}`.

```@example lq
using DifferentialGames, LinearAlgebra

n  = 4      # shared state dimension: [x, vx, y, vy]
dt = 0.1
T  = 3.0

A  = [1.0 dt 0.0 0.0;
      0.0 1.0 0.0 0.0;
      0.0 0.0 1.0 dt;
      0.0 0.0 0.0 1.0]

B  = [reshape([0.0, dt, 0.0, 0.0], n, 1),   # player 1 accelerates along x
      reshape([0.0, 0.0, 0.0, dt], n, 1)]   # player 2 accelerates along y

Q  = [diagm([1.0, 0.0, 0.0, 0.0]),          # player 1 cares about x
      diagm([0.0, 0.0, 1.0, 0.0])]          # player 2 cares about y
R  = [fill(0.01, 1, 1), fill(0.01, 1, 1)]
Qf = [10.0 * Q[1], 10.0 * Q[2]]
x0 = [2.0, 0.0, 2.0, 0.0]

game = LQGameProblem(A, B, Q, R, Qf, x0, T; dt=dt)
sol  = solve(game, FNELQ())

(converged = sol.converged, J₁ = get_cost(sol, 1), J₂ = get_cost(sol, 2))
```

### Extracting feedback gains

The strategy is a [`FeedbackStrategy`](@ref); `get_gain(strategy, i, k)` returns the ``m_i \times n`` gain of player ``i`` at step ``k`` (so that ``u_{i,k} = -K_{i,k}x_k``):

```@example lq
strategy = get_strategy(sol)
K1_0 = get_gain(strategy, 1, 1)
println("K₁ at t=0: ", round.(K1_0; digits=3))
```

### Verifying the equilibrium numerically

Rolling the closed loop forward with ``u_{i,k} = -K_{i,k}x_k`` must reproduce the stored state trajectory. This is a **check on the solution**, not a proof of optimality:

```@example lq
function simulate_closed_loop(A, B, strategy, x0, N)
    x = copy(x0)
    for k in 1:N
        u = [-(get_gain(strategy, i, k) * x)[1] for i in 1:length(B)]
        x = A * x + sum(B[i] * u[i] for i in eachindex(B))
    end
    return x
end

N   = size(sol.state_trajectory, 2) - 1
err = norm(simulate_closed_loop(A, B, strategy, x0, N) - sol.state_trajectory[:, end])
@assert err < 1e-8
err
```

## Time-Varying LQ Game

`LTVLQGameProblem` takes sequences: `A_seq[k]`, and for each player `B_seq[i][k]`, `Q_seq[i][k]`, `R_seq[i][k]`, plus one terminal matrix per player. `length(A_seq)` must equal `tf/dt`.

```@example lq
Nsteps = 30
A_seq = [Matrix(1.0I(4)) for _ in 1:Nsteps]
B_seq = [[Matrix(1.0I(4))[:, 1:1] for _ in 1:Nsteps],
         [Matrix(1.0I(4))[:, 2:2] for _ in 1:Nsteps]]
Q_seq = [[Matrix(1.0I(4)) for _ in 1:Nsteps] for _ in 1:2]
R_seq = [[fill(0.1, 1, 1) for _ in 1:Nsteps] for _ in 1:2]
Qf_ltv = [Matrix(1.0I(4)), Matrix(1.0I(4))]

ltv  = LTVLQGameProblem(A_seq, B_seq, Q_seq, R_seq, Qf_ltv, [1.0, 0.0, -1.0, 0.0], 3.0; dt=0.1)
sol_ltv = solve(ltv, FNELQ())
(converged = sol_ltv.converged, costs = get_costs(sol_ltv))
```

## Single-Player Limit (LQR)

A one-player game is classical LQR, which gives an independent check of the solver. The finite-horizon Riccati recursion gives ``J = \tfrac12 x_0^\top P_0 x_0``:

```@example lq
n_, m_ = 4, 2
lqr_game = LQGameProblem(
    Matrix(1.0I(n_)), [Matrix(1.0I(n_))[:, 1:m_]],
    [Matrix(1.0I(n_))], [Matrix(0.1I(m_))], [Matrix(10.0I(n_))],
    ones(n_), 2.0; dt=0.1,
)
sol_lqr = solve(lqr_game, FNELQ())

function lqr_cost(n, m, N, x0)
    Ad, Bd = Matrix(1.0I(n)), Matrix(1.0I(n))[:, 1:m]
    Qm, Rm = Matrix(1.0I(n)), Matrix(0.1I(m))
    P = Matrix(10.0I(n))
    for _ in 1:N
        K = (Rm + Bd' * P * Bd) \ (Bd' * P * Ad)
        P = Qm + Ad' * P * (Ad - Bd * K)
    end
    return 0.5 * x0' * P * x0
end

J_riccati = lqr_cost(n_, m_, 20, ones(n_))
@assert isapprox(get_cost(sol_lqr, 1), J_riccati; rtol=1e-8)
(FNELQ = get_cost(sol_lqr, 1), Riccati = J_riccati)
```

## What about `PDGNEProblem` with LQ costs?

`PDGNEProblem` stores separable *function* dynamics, so it is not recognised as linear: `FNELQ` rejects it (`FNELQ requires LinearDynamics`). Express LQ games through `LQGameProblem` / `LTVLQGameProblem`, or solve the `PDGNEProblem` with `iLQGames`.

## Further Reading

- [FNELQ solver reference](../solvers/fnelq.md) — solver options and implementation notes
- [LQGameProblem reference](../problem_types/lq_game.md) — full constructor API
- [Nonlinear Games tutorial](nonlinear_games.md) — extend to nonlinear dynamics with iLQGames
