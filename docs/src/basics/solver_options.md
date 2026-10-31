# Common Solver Options

Solver-specific options are keyword arguments of the **solver constructor**. The keyword arguments that every solver shares belong to `solve`.

## Arguments of `solve`

```julia
sol = solve(game, solver;
    verbose             = false,     # print iteration information (solver-dependent)
    warmstart           = nothing,   # a previous GNEPSolution or a WarmstartData
    check_compatibility = true,      # warn if the solver does not list the game's capabilities
)
```

`verbose` is **not** a field of `iLQGames`, `ALGAMES`, `LIBR`, `YiPavel` or `InverseLQGames`; passing `verbose=...` to those constructors raises a `MethodError`.

### Warmstarting

Warmstarting from the previous solution can reduce iteration counts when solving a sequence of related problems:

```julia
sol1  = solve(game1, iLQGames())
game2 = remake(game1; initial_state = x0_new)
sol2  = solve(game2, iLQGames(); warmstart = sol1)      # or WarmstartData(sol1)
```

### Capability checking

`solve` compares `required_capabilities(game)` with `solver_capabilities(typeof(solver))`. A mismatch only produces a `@warn`; a solver may still fail with an assertion (e.g. `iLQGames` on a constrained game, `FNELQ` on a game without `LinearDynamics`).

```julia
solver_capabilities(FNELQ)     # [:LQGame, :FeedbackPolicies, :UnconstrainedGame, :DiscreteTime]
solver_capabilities(iLQGames)  # [:NonlinearGame, :LQGame, :UnconstrainedGame, :DiscreteTime]
solver_capabilities(ALGAMES)   # [:LQGame, :NonlinearGame, :GNEP, :ConstrainedGame, :UnconstrainedGame,
                               #  :DiscreteTime, :SeparableDynamics, :SharedConstraints]
```

Symbols are drawn from the closed vocabulary `CANONICAL_CAPABILITIES`.

## Solver-Specific Options

The authoritative list for each solver is its struct docstring and its reference page:

| Solver | Constructor options |
|--------|---------------------|
| [`FNELQ`](../solvers/fnelq.md) | `check_singularity`, `rcond_threshold`, `regularization` |
| [`iLQGames`](../solvers/ilqgames.md) | `max_iter`, `ε_conv`, `β`, `η_min`, `max_state_step`, `μ_init`, `μ_max`, `μ_scale`, `μ_decay`, `discretization` |
| [`ALGAMES`](../solvers/algames.md) | `outer_iter`, `ρ_init`, `ρ_increase`, `ρ_max`, `inner_iter`, `reg`, `ls_iter`, `ls_beta`, `ls_tau`, `tol_opt`, `tol_dyn`, `tol_con`, `reset_duals`, `terminal_constraints` |
| [`LIBR`](../solvers/libr.md) | `max_iter`, `tol`, `inner_iter`, `step_size`, `ls_beta`, `ls_max_iter`, `ls_armijo`, `col_penalty`, `col_slack` |
| [`YiPavel`](../solvers/yipavel.md) | `max_iter`, `tol`, `τ`, `ν`, `σ` |
| [`InverseLQGames`](../solvers/invlq.md) | `tol`, `K_ridge`, `svd_tol` |

For which solver to use, see the [Solver Selection Guide](solver_selection.md).
