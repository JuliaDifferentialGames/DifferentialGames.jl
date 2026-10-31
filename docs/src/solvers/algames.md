# ALGAMES

**Augmented Lagrangian GAMES** solver. Computes a local open-loop Nash equilibrium for finite-horizon nonlinear games with shared and/or private constraints using an augmented Lagrangian outer loop.

## Algorithm

ALGAMES combines augmented Lagrangian constraint handling with a Newton-based inner loop:

1. **Outer loop** — maintain a penalty parameter ``\rho`` and dual variables ``\lambda`` for each constraint. At each outer iteration, update ``\lambda`` and scale up ``\rho``.
2. **Inner loop** — treat the augmented Lagrangian (penalized) problem as an unconstrained game. Solve to approximate Nash stationarity using a Newton step on the KKT system.
3. **Convergence** — stop when both primal feasibility (constraint violation) and dual stationarity residuals fall below their respective tolerances.

The result is an **open-loop** Nash equilibrium: each player commits to a full control sequence ``u_i = (u_i^0, \ldots, u_i^{N-1})`` simultaneously.

!!! warning "What is (and is not) guaranteed"
    ALGAMES drives a penalized KKT system toward stationarity and reports residuals. The original paper argues that the multipliers of a shared constraint can be taken equal across players, which corresponds to a *normalized* (variational) generalized Nash equilibrium. This documentation does **not** claim that every point returned by this implementation is a normalized Nash equilibrium: the solver is a local method, it can stop with `converged = false`, and the claim has not been verified against this implementation (see the audit). Verify the constraints and the residuals in `sol.solver_info` for your problem.

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
    terminal_constraints = false,
))

# `verbose` and `warmstart` are arguments of `solve`, not of the solver:
sol = solve(game, ALGAMES(); verbose=true)
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

### Terminal constraints
| Option | Default | Description |
|--------|---------|-------------|
| `terminal_constraints` | `false` | Also enforce state-only constraint rows at the terminal state `x(N+1)` |

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
| `LQGameProblem` | Accepted (`:LQGame` is a declared capability); `FNELQ` is exact and faster |
| Nonlinear dynamics / costs | Yes |

## Warmstarting

ALGAMES supports warmstarting from a prior solution. This is particularly effective when solving a sequence of problems in a receding-horizon loop:

```julia
sol1 = solve(game1, ALGAMES())

game2 = remake(game1; initial_state = x0_new)
sol2  = solve(game2, ALGAMES(); warmstart=sol1)   # a GNEPSolution or a WarmstartData
```

When `reset_duals=false`, the solver will reuse dual variables (λ, ρ, μ) from the warmstart data.

## Practical Tips

**Tuning penalty growth**: If ALGAMES fails to converge, try reducing `ρ_increase` (e.g., to 2.0) to grow the penalty more conservatively, or increase `reg` to stabilize the Newton solve.

**Warmstart from iLQGames**: for a nonlinear constrained problem you can solve the unconstrained version (`game_unc`, same players without the constraints) with iLQGames first and pass the result as a warmstart to ALGAMES (the combination is accepted by the API; its benefit is problem-dependent and not benchmarked here):

```julia
sol_ilq = solve(game_unc, iLQGames())
sol_alg = solve(game, ALGAMES(); warmstart=sol_ilq)
```

**Open-loop vs feedback**: ALGAMES produces open-loop strategies. If you need a feedback (closed-loop) policy, iLQGames is the better choice for unconstrained problems.

## Constraint enforcement points and terminal constraints

By default constraints are enforced on the pairs ``(x_k, u_k)`` for ``k = 1,\dots,N``, and `con_vio` is computed over the same nodes. The **terminal state** ``x_{N+1}`` is then unconstrained, so a shared constraint can be violated at the last node while `converged = true` and `con_vio ≈ 0`.

Set `ALGAMES(; terminal_constraints = true)` to also enforce the constraints at ``x_{N+1}``. The terminal state has no control, so a constraint row is enforced there only if it does **not** depend on ``u`` (detected by evaluating the constraint at two different controls); control-dependent rows such as `control_bounds` apply to nodes ``1,\dots,N`` only. The terminal node gets its own multipliers and penalties and is included in `con_vio`, the dual ascent and `sol.solver_info[:dual_variables]` (whose `λ`, `ρ` then have length `nc_step·(N+1)`). Example (goals 1.0 apart with a 1.5 separation constraint): without the option the final distance is 1.1 and `con_vio = 0`; with it the final distance is 1.5 (covered by a regression test).

## Convergence

Convergence is declared when both residuals drop below their tolerances:

```math
\text{opt\_vio} < \varepsilon_{\text{opt}}, \quad \text{dyn\_vio} < \varepsilon_{\text{dyn}}, \quad \text{con\_vio} < \varepsilon_{\text{con}}
```

where `opt_vio` is the (normalized) stationarity residual, `dyn_vio` the dynamics residual and `con_vio` the maximum constraint violation. Their histories are stored in `sol.solver_info[:opt_vio]`, `[:dyn_vio]` and `[:con_vio]` (also `[:history]`, `[:dual_variables]`).

**Interpreting a `false` flag.** In a two-vehicle test the default tolerances ended at `con_vio ≈ 2·10⁻⁷` but `opt_vio ≈ 116` after 50 outer iterations, so the trajectory satisfied the constraints while the stationarity test failed; loosening `tol_opt` to `1e-2` produced `converged = true` after 3 iterations (a different local solution). Results are sensitive to the options (`ρ_init`, `inner_iter`, `tol_opt`), so check the constraints independently, as in the [Constrained Games tutorial](../tutorials/constrained_games.md).

## Known limitations

- Local method; sensitive to `ρ_init`, `inner_iter`, `tol_opt` (different settings can reach different local solutions of the same problem).
- Constraints are enforced at nodes `1:N` only unless `terminal_constraints = true`; even then, control-dependent rows are not enforced at the terminal node.
- Open-loop strategies only (no feedback gains).
- With the default `tol_opt` the stationarity test can fail while constraints are satisfied (see *Interpreting a `false` flag* above).

## References

[LeCleach2022](@cite) ([arXiv:2104.08452](https://arxiv.org/abs/2104.08452)).

See also the implementation notes in [DifferentialGamesBaseSolvers.jl](https://github.com/JuliaDifferentialGames/DifferentialGamesBaseSolvers.jl).
