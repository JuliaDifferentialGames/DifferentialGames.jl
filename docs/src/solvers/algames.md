# ALGAMES

**Augmented Lagrangian GAMES** solver. Computes a local open-loop Nash equilibrium for finite-horizon nonlinear games with shared and/or private constraints using an augmented Lagrangian outer loop.

## Algorithm

ALGAMES combines augmented Lagrangian constraint handling with a Newton-based inner loop:

1. **Outer loop** — maintain a penalty parameter ``\rho`` and dual variables ``\lambda`` for each constraint. At each outer iteration, update ``\lambda`` and scale up ``\rho``.
2. **Inner loop** — treat the augmented Lagrangian (penalized) problem as an unconstrained game. Solve to approximate Nash stationarity using a Newton step on the KKT system.
3. **Convergence** — stop when both primal feasibility (constraint violation) and dual stationarity residuals fall below their respective tolerances.

The result is an **open-loop** Nash equilibrium: each player commits to a full control sequence ``u_i = (u_i^0, \ldots, u_i^{N-1})`` simultaneously.

## Usage

```julia
sol = solve(game, ALGAMES())

# With options:
sol = solve(game, ALGAMES(;
    max_outer  = 30,
    max_inner  = 100,
    ρ_init     = 1.0,
    ρ_scale    = 10.0,
    ρ_max      = 1e6,
    ε_primal   = 1e-4,
    ε_dual     = 1e-4,
    reg        = 1e-3,
    verbose    = false,
))
```

```@docs
ALGAMES
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `max_outer` | `30` | Maximum augmented Lagrangian outer iterations |
| `max_inner` | `100` | Maximum Newton iterations per outer loop |
| `ρ_init` | `1.0` | Initial penalty parameter |
| `ρ_scale` | `10.0` | Penalty growth factor per outer iteration |
| `ρ_max` | `1e6` | Maximum penalty (caps penalty growth) |
| `ε_primal` | `1e-4` | Primal feasibility tolerance (constraint violation) |
| `ε_dual` | `1e-4` | Dual feasibility tolerance (stationarity residual) |
| `reg` | `1e-3` | Regularization added to Newton Hessian |
| `verbose` | `false` | Print per-iteration log |

## Applicable Problem Types

| Problem Type | Supported |
|--------------|-----------|
| `PDGNEProblem` (unconstrained) | Yes |
| `PDGNEProblem` (shared constraints) | Yes |
| `PDGNEProblem` (private constraints) | Yes |
| `LQGameProblem` | No — use `FNELQ` |
| Nonlinear dynamics / costs | Yes |

## Warmstarting

ALGAMES supports warmstarting from a prior solution. This is particularly effective when solving a sequence of problems in a receding-horizon loop:

```julia
sol1 = solve(game1, ALGAMES())

game2 = remake(game1; initial_state = x0_new)
ws    = WarmstartData(sol1)
sol2  = solve(game2, ALGAMES(); warmstart=ws)
```

## Practical Tips

**Tuning penalty growth**: If ALGAMES fails to converge, try reducing `ρ_scale` (e.g., to 2.0) to grow the penalty more conservatively, or increase `reg` to stabilize the Newton solve.

**Warmstart from iLQGames**: For nonlinear constrained problems, solve the unconstrained version with iLQGames first to get a good initial trajectory, then pass it as a warmstart to ALGAMES:

```julia
sol_ilq = solve(game, iLQGames())
sol_alg = solve(game, ALGAMES(); warmstart=WarmstartData(sol_ilq))
```

**Open-loop vs feedback**: ALGAMES produces open-loop strategies. If you need a feedback (closed-loop) policy, iLQGames is the better choice for unconstrained problems.

## Convergence

Convergence is declared when both residuals drop below their tolerances:

```math
\|g_{\text{primal}}\|_\infty < \varepsilon_{\text{primal}}, \quad \|g_{\text{dual}}\|_\infty < \varepsilon_{\text{dual}}
```

where ``g_{\text{primal}}`` is the constraint violation and ``g_{\text{dual}}`` is the stationarity residual of the augmented Lagrangian with respect to the primal variables.
