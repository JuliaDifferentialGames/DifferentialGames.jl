# iLQGames

**Iterative Linear Quadratic Games** solver. Computes a local feedback Nash equilibrium for finite-horizon nonlinear games by repeatedly linearizing the dynamics and solving the resulting LQ subgame.

## Algorithm

iLQGames extends the iterative LQR (iLQR) idea to the multi-player setting. Each outer iteration consists of:

1. **Forward pass** — roll out the current strategy from the initial state to obtain a nominal trajectory ``(\bar{x}_0, \ldots, \bar{x}_N)``.
2. **Linearization** — compute first-order Taylor expansions of each player's dynamics and second-order expansions of each player's cost along the nominal trajectory.
3. **LQ subgame** — solve the resulting time-varying LQ game exactly with FNELQ to obtain gain corrections ``\delta K_i^k``.
4. **Line search** — step from the nominal trajectory toward the corrected trajectory, backtracking until a sufficient decrease in the total Nash residual is found.

Convergence is declared when the maximum change in any element of the trajectory falls below ``\varepsilon_\text{conv}`` (default `0.05`, an absolute tolerance on states).

## Usage

```julia
sol = solve(game, iLQGames())

# With options:
sol = solve(game, iLQGames(;
    max_iter          = 200,
    ε_conv            = 0.05,
    β                 = 0.5,
    η_min             = 0.5^20,
    max_state_step    = 1.0,
    μ_init            = 1.0,
    μ_max             = 1e6,
    μ_scale           = 10.0,
    μ_decay           = 0.5,
    discretization    = ZOHDiscretization(),
))

# `verbose` and `warmstart` are arguments of `solve`, not of the solver:
sol = solve(game, iLQGames(); verbose=true)
```

## Solver Type Documentation

```@docs
iLQGames
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `max_iter` | `200` | Maximum number of outer iterations |
| `ε_conv` | `0.05` | Trajectory-change convergence threshold |
| `β` | `0.5` | Line search backtrack factor |
| `η_min` | `0.5^20` | Minimum line search step |
| `max_state_step` | `1.0` | Maximum state step accepted in line search |
| `μ_init` | `1.0` | Initial S-regularization |
| `μ_max` | `1e6` | Maximum regularization |
| `μ_scale` | `10.0` | Regularization growth factor on ill-conditioning |
| `μ_decay` | `0.5` | Regularization decay factor on accepted step |
| `discretization` | `ZOHDiscretization()` | Method for auto-discretising continuous dynamics |

## Capabilities

```@docs
solver_capabilities(::Type{iLQGames})
```

## Applicable Problem Types

| Problem Type | Supported |
|--------------|-----------|
| `LQGameProblem` | Yes — but it iterates and is regularized; prefer `FNELQ` |
| `LTVLQGameProblem` | Yes |
| `PDGNEProblem` | Yes, unconstrained only |
| Nonlinear dynamics / costs | Yes |
| Shared / private constraints | No — use `ALGAMES` |

## Convergence

iLQGames finds a **local** feedback Nash equilibrium of successive LQ approximations. There is no guarantee of global optimality, and the solution found may depend on the initial strategy (zero by default). The trajectory-change test is a numerical stopping criterion, **not** a certificate of equilibrium. Always inspect `sol.converged`, `sol.iterations` and `sol.solver_info[:trajectory_change_history]`.

On an exactly linear-quadratic problem the converged cost approaches the exact feedback-Nash cost from `FNELQ`; the rate is governed by the regularization `μ_init` (default `1.0`), so a small `ε_conv` is needed for tight agreement (a double-integrator `PDGNEProblem` with exact per-player cost 7.9835 is covered by a regression test).

The primary output is a `FeedbackStrategy` from the last converged FNELQ solve, which constitutes a feedback Nash equilibrium of the LQ subgame built around the converged nominal trajectory (§IV-B of Fridovich-Keil et al. 2020).

Convergence criterion:

```math
\max_k \|x_k^{(j+1)} - x_k^{(j)}\|_\infty < \varepsilon_{\text{conv}}
```

## Assumptions and guarantees

None beyond local smoothness. The returned point is a feedback-Nash equilibrium of the *last LQ approximation*, not of the nonlinear game; the trajectory-change test is a stopping rule. Results depend on the initial (zero-control) trajectory, the horizon and the regularization `μ_init`.

## Known limitations

- No constraints (asserts); use [ALGAMES](algames.md).
- Local method; different initializations can give different solutions.
- Linear convergence governed by the regularization `μ` (default `1.0`): tight agreement with an exact LQ solution needs a small `ε_conv`.
- Dynamics and costs must be ForwardDiff-compatible.

## Notes

- Dynamics and stage costs must be differentiable with respect to state and control. The solver uses `ForwardDiff.jl` for automatic differentiation; closures must be ForwardDiff-compatible (avoid type-specific branches or non-differentiable operations).
- For **constrained** games, use [`ALGAMES`](algames.md) instead, optionally warmstarted from an iLQGames solution.
- For **LQ** games, [`FNELQ`](fnelq.md) is faster (direct solver, no iteration).

## References

[FridovichKeil2020](@cite) ([arXiv:1909.04694](https://arxiv.org/abs/1909.04694))
