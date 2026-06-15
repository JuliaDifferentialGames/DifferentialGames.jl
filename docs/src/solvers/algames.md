# ALGAMES

**Augmented Lagrangian GAMES** solver. Computes a local open-loop Nash equilibrium for finite-horizon nonlinear games with shared and/or private constraints using an augmented Lagrangian outer loop.

## Algorithm

ALGAMES combines augmented Lagrangian constraint handling with a Newton-based inner loop:

1. **Outer loop** — maintain a penalty parameter ``\rho`` and dual variables ``\lambda`` for each constraint. At each outer iteration, update ``\lambda`` and scale up ``\rho``.
2. **Inner loop** — treat the augmented Lagrangian (penalized) problem as an unconstrained game. Solve to approximate Nash stationarity using a Newton step on the KKT system.
3. **Convergence** — stop when both primal feasibility (constraint violation) and dual stationarity residuals fall below their respective tolerances.

The result is an **open-loop** Nash equilibrium: each player commits to a full control sequence ``u_i = (u_i^0, \ldots, u_i^{N-1})`` simultaneously.

With shared inequality constraints, the solver converges to a **Normalized Nash Equilibrium (NNE)** because identical dual-ascent updates enforce equal multipliers on shared constraints.

## Usage

```julia
sol = solve(game, ALGAMES())

# With options:
sol = solve(game, ALGAMES(;
    outer_iter  = 50,
    ρ_init      = 1.0,
    ρ_increase  = 10.0,
    ρ_max       = 1e8,
    inner_iter  = 10,
    reg         = 1e-3,
    ls_iter     = 20,
    ls_beta     = 0.1,
    ls_tau      = 0.5,
    tol_opt     = 1e-4,
    tol_dyn     = 1e-4,
    tol_con     = 1e-3,
    reset_duals = true,
    verbose     = false,
))
```

## Solver Type Documentation

```@docs
ALGAMES
```

## Options

### Outer AL Loop
| Option | Default | Description |
|--------|---------|-------------|
| `outer_iter` | `50` | Maximum augmented Lagrangian iterations |
| `ρ_init` | `1.0` | Initial penalty weight |
| `ρ_increase` | `10.0` | Geometric multiplier for penalty growth |
| `ρ_max` | `1e8` | Cap on penalty parameter |

### Newton Inner Loop
| Option | Default | Description |
|--------|---------|-------------|
| `inner_iter` | `10` | Maximum Newton steps per outer iteration |
| `reg` | `1e-3` | Tikhonov regularization on Hessian |

### Line Search
| Option | Default | Description |
|--------|---------|-------------|
| `ls_iter` | `20` | Maximum backtracks in Armijo line search |
| `ls_beta` | `0.1` | Sufficient-decrease fraction |
| `ls_tau` | `0.5` | Step contraction factor |

### Convergence Tolerances
| Option | Default | Description |
|--------|---------|-------------|
| `tol_opt` | `1e-4` | Stationarity norm tolerance |
| `tol_dyn` | `1e-4` | Dynamics residual tolerance |
| `tol_con` | `1e-3` | Maximum constraint violation tolerance |

### Warm-start
| Option | Default | Description |
|--------|---------|-------------|
| `reset_duals` | `true` | If false, load dual variables from WarmstartData |

## Capabilities

```@docs
solver_capabilities(::Type{ALGAMES})
```

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

When `reset_duals=false`, the solver will reuse dual variables (λ, ρ, μ) from the warmstart data.

## Practical Tips

**Tuning penalty growth**: If ALGAMES fails to converge, try reducing `ρ_increase` (e.g., to 2.0) to grow the penalty more conservatively, or increase `reg` to stabilize the Newton solve.

**Warmstart from iLQGames**: For nonlinear constrained problems, solve the unconstrained version with iLQGames first to get a good initial trajectory, then pass it as a warmstart to ALGAMES:

```julia
sol_ilq = solve(game, iLQGames())
sol_alg = solve(game, ALGAMES(); warmstart=WarmstartData(sol_ilq))
```

**Open-loop vs feedback**: ALGAMES produces open-loop strategies. If you need a feedback (closed-loop) policy, iLQGames is the better choice for unconstrained problems.

## Convergence

Convergence is declared when both residuals drop below their tolerances:

```math
\|g_{\text{primal}}\|_\infty < \varepsilon_{\text{opt}}, \quad \|g_{\text{dyn}}\|_\infty < \varepsilon_{\text{dyn}}, \quad \text{max constraint violation} < \varepsilon_{\text{con}}
```

where ``g_{\text{opt}}`` is the stationarity residual and ``g_{\text{dyn}}`` is the dynamics residual.

## References

Le Cleac'h, S., Schwager, M., Manchester, I. (2021). *ALGAMES: A Fast Augmented Lagrangian Solver for Constrained Dynamic Games.* [arXiv:2104.08452](https://arxiv.org/abs/2104.08452).

See also the implementation notes in [DifferentialGamesBaseSolvers.jl](https://github.com/JuliaDifferentialGames/DifferentialGamesBaseSolvers.jl).
