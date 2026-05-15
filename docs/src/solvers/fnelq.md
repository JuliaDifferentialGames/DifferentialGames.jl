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
```

FNELQ takes no tunable parameters — it is a direct solver with no iteration.

```@docs
FNELQ
```

## Applicable Problem Types

| Problem Type | Supported |
|--------------|-----------|
| `LQGameProblem` | Yes |
| `LTVLQGameProblem` | Yes |
| `PDGNEProblem` (LQ costs) | Yes |
| Nonlinear dynamics / costs | No |
| Constrained games | No |

## When to Use

- Any LQ game where you want the exact solution in one shot.
- Inner loop of an inverse game solver (fastest per-call cost).
- Validating that nonlinear solvers converge to the right answer on LQ problems.

FNELQ is always the right choice when the problem is LQ. It does not iterate, does not require tuning, and produces the exact feedback Nash equilibrium up to floating-point precision.

## Complexity

``O(N \cdot N_{\text{steps}} \cdot n^3)`` where ``n`` is the total state dimension. The dominant cost is the per-step matrix solve for the Nash gains.
