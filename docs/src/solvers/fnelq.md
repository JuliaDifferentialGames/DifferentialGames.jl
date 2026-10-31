# FNELQ

**Feedback Nash Equilibrium Linear Quadratic** solver. Computes the exact, closed-form feedback Nash equilibrium for finite-horizon discrete-time LQ games in a single backward pass.

## Algorithm

FNELQ solves the coupled Riccati equations that arise from the Nash optimality conditions. For an N-player game with dynamics

```math
x_{k+1} = A x_k + \sum_{i=1}^N B_i u_i^k
```

and quadratic costs, the feedback Nash strategies have the form

```math
u_i^k = -K_i^k x_k
```

where the gain matrices ``K_i^k`` are computed by solving the backward Riccati recursion:

```math
V_i^k = Q_i^k + \sum_j (K_j^k)^\top R_{ij}^k K_j^k + (A - \sum_j B_j K_j^k)^\top V_i^{k+1} (A - \sum_j B_j K_j^k)
```

At each step the gains are found by simultaneously solving the N-player coupled linear system arising from the stationarity conditions, which is dense but small (``\sum_i m_i \times \sum_i m_i``).

## Usage

```julia
sol = solve(game, FNELQ())

# With options:
sol = solve(game, FNELQ(;
    check_singularity = true,
    rcond_threshold   = 1e-10,
    regularization     = 0.0
))
```

## Solver Type Documentation

```@docs
FNELQ
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `check_singularity` | `true` | Warn when the Nash gain matrix is ill-conditioned |
| `rcond_threshold` | `1e-10` | Reciprocal condition number below which a warning is issued |
| `regularization` | `0.0` | Tikhonov regularization added to the gain system |

## Capabilities

```@docs
solver_capabilities(::Type{FNELQ})
```

## Applicable Problem Types

| Problem Type | Supported |
|--------------|-----------|
| `LQGameProblem` | Yes |
| `LTVLQGameProblem` | Yes |
| `PDGNEProblem` (LQ costs) | **No** — `PDGNEProblem` stores function dynamics; FNELQ asserts `FNELQ requires LinearDynamics` |
| Nonlinear dynamics / costs | No |
| Constrained games | No |

## When to Use

- Any LQ game where you want the exact solution in one shot.
- Inner loop of an inverse game solver (fastest per-call cost).
- Validating that nonlinear solvers converge to the right answer on LQ problems.

When the problem can be written as an `LQGameProblem` / `LTVLQGameProblem`, FNELQ is the natural choice: it does not iterate, needs no tuning, and computes the feedback-Nash gains from the coupled Riccati recursion up to floating-point accuracy (provided the per-stage gain systems are non-singular; see `check_singularity`).

## Reported costs

Reported player costs include the factor ``\tfrac12``: ``J_i = \tfrac12\big[\sum_k x_k^\top Q_i x_k + u_{i,k}^\top R_i u_{i,k} + x_N^\top Q_{f,i}x_N\big]``. For a one-player game this equals ``\tfrac12 x_0^\top P_0 x_0`` from the standard Riccati recursion; the [LQ tutorial](../tutorials/lq_games.md) checks this numerically.

## Diagnostics and interpretation

- `sol.converged` is `true` when the backward pass completed.
- `sol.solver_info` contains `:min_rcond`, `:rcond_history`, `:max_P_norm`, `:cost_to_go_matrices`.
- A small `:min_rcond` signals a nearly singular gain system; the equilibrium may not be unique or numerically reliable.
- `get_strategy(sol)` is a `FeedbackStrategy`: `get_gain(strategy, i, k)` returns ``K_{i,k}`` with ``u_{i,k} = -K_{i,k}x_k``.

## Complexity

``O(N \cdot N_{\text{steps}} \cdot n^3)`` where ``n`` is the total state dimension. The dominant cost is the per-step matrix solve for the Nash gains.

## Assumptions and guarantees

For a non-singular stage system ``S_k`` the recursion returns the unique feedback Nash strategies of the LQ game ([Optimality Conditions](../math/optimality_conditions.md)). Existence/uniqueness can fail when some ``S_k`` is singular; FNELQ then warns and sets `converged = false`. Verified here: single-player limit equals the Riccati LQR value, and the closed loop reproduces the stored trajectory.

## Known limitations

- Requires `LinearDynamics` (`LQGameProblem`/`LTVLQGameProblem`); `PDGNEProblem` is rejected.
- No constraints.
- Discrete time only.
- The affine (linear-term) recursion used inside iLQGames had a sign error before the fix recorded in the audit; with current sources it is covered by a regression test.

## References

The algorithm is the standard feedback Nash equilibrium solution for discrete-time LQ games [BasarOlsder1999](@cite). See also the implementation notes in [DifferentialGamesBaseSolvers.jl](https://github.com/JuliaDifferentialGames/DifferentialGamesBaseSolvers.jl).
