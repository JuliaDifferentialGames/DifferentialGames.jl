# Common Solver Options

All solvers in DifferentialGames.jl accept a common set of keyword arguments alongside their solver-specific options.

## Universal Options

```julia
sol = solve(game, solver;
    verbose   = false,         # print iteration log
    warmstart = nothing,       # WarmstartData from a prior solve
)
```

### Verbosity

With `verbose=true`, solvers print a per-iteration summary:

```
[iter   1] Δ=1.243e-02  J₁=4.321  J₂=3.876
[iter   2] Δ=3.847e-04  J₁=4.298  J₂=3.854
...
✓ converged in 12 iterations (0.043 s)
```

### Warmstarting

If you are solving a sequence of related problems (e.g., in a receding-horizon loop), warmstarting from the previous solution can dramatically reduce iteration count:

```julia
sol1 = solve(game1, iLQGames())

# Perturb the problem slightly
game2 = remake(game1; initial_state = x0 + 0.01 * randn(n))

ws  = WarmstartData(sol1)
sol2 = solve(game2, iLQGames(); warmstart=ws)
```

## Solver-Specific Options

### FNELQ

FNELQ has no tunable parameters — it is an exact direct solver.

```julia
sol = solve(game, FNELQ())
```

**Applicable to:** `LQGameProblem`, `LTVLQGameProblem`, `PDGNEProblem` with LQ costs

### iLQGames

```julia
sol = solve(game, iLQGames(;
    max_iter   = 100,     # maximum iterations
    ε_conv     = 1e-6,    # convergence threshold on max trajectory change
    α_min      = 1e-4,    # minimum line-search step size
    verbose    = false,
))
```

**Convergence criterion:** ``\max_k \|x_{k}^{(j+1)} - x_{k}^{(j)}\|_\infty < \varepsilon_\text{conv}``

### ALGAMES

```julia
sol = solve(game, ALGAMES(;
    max_outer   = 30,      # max augmented Lagrangian outer iterations
    max_inner   = 100,     # max Newton iterations per outer loop
    ρ_init      = 1.0,     # initial penalty parameter
    ρ_scale     = 10.0,    # penalty growth rate per outer iteration
    ρ_max       = 1e6,     # penalty cap
    ε_primal    = 1e-4,    # primal feasibility tolerance
    ε_dual      = 1e-4,    # dual feasibility tolerance
    reg         = 1e-3,    # regularization on Newton Hessian
    verbose     = false,
))
```

**Convergence criterion:** both primal (constraint violation) and dual (stationarity) residuals below their respective tolerances.

## Solver Capabilities

Use `solver_capabilities` to check what problem types a solver supports:

```julia
solver_capabilities(FNELQ)
# → [:lq_game, :ltv_lq_game, :feedback_nash, :unconstrained]

solver_capabilities(iLQGames)
# → [:nonlinear, :feedback_nash, :unconstrained]

solver_capabilities(ALGAMES)
# → [:nonlinear, :open_loop_nash, :constrained, :unconstrained]
```

## Choosing a Solver

| Scenario | Recommended solver |
|----------|-------------------|
| LQ game (exact solution needed) | `FNELQ` |
| Nonlinear game, no constraints | `iLQGames` |
| Constrained game (private or shared) | `ALGAMES` |
| Nonlinear + constraints, warm-startable | `iLQGames` → `ALGAMES` |
| Inner loop of an inverse solver | `FNELQ` (fastest per call) |

For nonlinear constrained games, a practical approach is to first solve with iLQGames (unconstrained) to get a good initial trajectory, then refine with ALGAMES using that trajectory as a warmstart.
