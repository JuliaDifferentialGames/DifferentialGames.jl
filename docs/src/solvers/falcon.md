# FALCON

**Fast Augmented Lagrangian Convexification for Open-loop Nash equilibria** ([OutlandArya](@cite)). Computes a local open-loop generalized Nash equilibrium of a non-convex *partially-decoupled* GNEP (`PDGNEProblem`): players with their own nonlinear dynamics, coupled through constraints and objectives such as collision avoidance, communication range or competition for progress. FALCON is a separate package, [`FALCON.jl`](https://github.com/BennetOutland/FALCON.jl), and a *soft dependency* of `DifferentialGamesBaseSolvers.jl`.

!!! note "Status"
    The solver lives in its own repository and follows the `GameSolver` interface of `DifferentialGamesBase.jl`. The examples on this page are not executed by the documentation build, because FALCON is not a dependency of the documentation environment.

## Installing and loading

FALCON is not in the General registry, and `DifferentialGamesBaseSolvers.jl` does not depend on it. Install both, then load FALCON *after* the solvers package so that the extension `DifferentialGamesBaseSolversFALCONExt` activates:

```julia
using Pkg
Pkg.add(url = "https://github.com/BennetOutland/FALCON.jl")

using DifferentialGamesBase, DifferentialGamesBaseSolvers
using FALCON                                   # activates the extension
```

Without `using FALCON`, [`falcon`](@ref) has no methods and raise an error that says what to load. Everything else in `DifferentialGamesBaseSolvers.jl` is unaffected, and FALCON's own dependencies (Clarabel, ForwardDiff, SCS, Plots, …) are never installed for users who do not ask for it.

You can use FALCON directly, without the extension: `FALCON.FALCONSolver` is an ordinary `GameSolver`.

## Algorithm

Each outer iteration of FALCON

1. **convexifies** every player's problem around the current trajectories: dynamics and non-convex constraints are linearized, and the coupling is handled by slack variables penalized with an augmented Lagrangian of weight ``\omega``;
2. adds a **proximal term** (weight ``\eta``) and a **trust region** of radius ``r`` on the step;
3. solves all players' convexified problems at once. Because the convexified game is a *potential game*, its equilibrium is the minimizer of one joint second-order cone program (SOCP), solved with [Clarabel](https://clarabel.org);
4. compares the actual decrease ``\Delta Q`` of the players' augmented Lagrangians with the decrease predicted by the convex model, accepts or rejects the step, and grows or shrinks ``r`` accordingly;
5. increases ``\omega`` (by the factor ``\beta``, up to ``\omega_{\max}``) and updates the multipliers when the infeasibility ``\chi`` has not decreased enough.

The solver stops when the players' predicted Nash decrease ``\max_i|\Delta Q^i|\le\epsilon_{\text{opt}}`` and the infeasibility ``\chi\le\epsilon_{\text{feas}}`` hold together. Because there is one SOCP per iteration and no per-player best-response loop, the cost of an iteration does not grow with the number of inner iterations of a Nash solver. The convergence theory (global convergence of the sequential convexification to a stationary point of the penalized problem) is in the paper.

!!! warning "What is guaranteed"
    FALCON is a local method: it needs a warm-start trajectory for every player, and the equilibrium it returns depends on it. `converged = true` means the two tolerances above were met; it is not a certificate of global optimality. Every convexified solve is also checked against the closed-form minimizer of the coupling slack (the *ζ certificate*); a solve that fails the check is rejected.

## Usage

```julia
using DifferentialGamesBase, DifferentialGamesBaseSolvers, FALCON

game   = ...                                    # a PDGNEProblem
solver = falcon(; ϵ_opt = 2e-2, ϵ_feas = 1e-4)  # = FALCON.FALCONSolver(; ...)
ws     = ...                                    # WarmstartData: an initial trajectory per player
sol    = solve(game, solver; warmstart = ws)

sol.converged
sol.iterations                                  # outer (SCP) iterations
sol.solver_info[:infeasibility]                 # χ
sol.solver_info[:ΔQ_max]                        # max_i |ΔQᵢ|
```

`sol` is a `GNEPSolution`; its `solver_info` also holds the cost, feasibility and trust-region histories, the final penalty weight and radii, the multipliers, and an iteration breakdown (accepted and rejected steps).

The solver supports problems tagged `:GNEP :NonlinearGame :SeparableDynamics :SharedConstraints :ConstrainedGame :PD_GNEP :DiscreteTime`; see [Choosing a solver](../basics/solver_selection.md).

### Parameters

The defaults of `FALCONSolver` are conservative. The most important keywords:

| Keyword | Default | Meaning |
|---------|---------|---------|
| `ϵ_opt`, `ϵ_feas` | `1e-4`, `1e-4` | tolerances on ``\max_i\lvert\Delta Q^i\rvert`` and on the infeasibility ``\chi`` |
| `max_iterations` | 100 | outer iteration limit |
| `r0`, `r_min`, `r_max` | `0.1`, `1e-6`, `10` | initial, minimum and maximum trust-region radius |
| `ρ_0`, `ρ_1`, `ρ_2`, `α_1`, `α_2` | `0`, `0.25`, `0.70`, `2`, `3` | step-acceptance thresholds and trust-region shrink/grow factors |
| `w0`, `β`, `ω_max` | `1`, `2`, `1e3` | initial augmented-Lagrangian weight, its growth factor, its cap |
| `η` | `1` | proximal weight |
| `γ_δ` | `0.5` | decay of the dual-update trigger |
| `ζ_cert_abs_tol`, `ζ_cert_rel_tol`, `ζ_cert_polish_ΔJ_tol` | `5e-4`, `1e-3`, `1e-2` | tolerances of the ζ certificate |
| `infeasibility_preprocessing`, `preprocess_iters` | `false`, `3` | project an infeasible warm start towards feasibility first |
| `zero_sum` | `false` | step-acceptance ratio for exactly zero-sum games |
| `scaling` | `nothing` | per-player diagonal variable scaling (`TrajectoryScaling`) |
| `inner_solver`, `linear_solver` | `:clarabel`, `:qdldl` | SOCP backend (`:scs` is experimental) and its linear solver |
| `socp_tol_loose`, `socp_tol_tight`, `socp_tol_switch` | `1e-6`, `1e-6`, `1e-2` | SOCP tolerances and the switch between them |
| `diagnostics` | `false` | record the per-iteration ``\Delta Q`` / ``\Delta L`` decomposition |

### Tuning

A few limits matter on hard problems; they were identified on a racing game with sharp corners and hold more generally.

- **Penalty cap.** If ``\omega`` sits at ``\omega_{\max}`` while ``\chi`` falls only slowly, the cap is too low for the coupling constraint: raise `ω_max` (for example to `1e4`).
- **Trust-region floor.** With a large penalty the radius can shrink to `r_min` with rejected steps; lowering `r_min` (for example to `1e-5`) lets it recover.
- **ζ certificate.** If most convexified solves are rejected by the certificate, relax `ζ_cert_abs_tol`, `ζ_cert_rel_tol` and `ζ_cert_polish_ΔJ_tol`.
- **Infeasible warm starts.** Turn on `infeasibility_preprocessing` when the initial trajectories violate the coupling constraints.

Tune on your own problem; settings tuned for one game are a starting point, not a guarantee.

## Capabilities

`solver_capabilities(FALCON.FALCONSolver)` is `[:GNEP, :NonlinearGame, :SeparableDynamics, :SharedConstraints, :ConstrainedGame, :PD_GNEP, :DiscreteTime]`. FALCON needs a `PDGNEProblem` (separable dynamics) and does not accept a coupled-dynamics `GameProblem`.

## Limitations

- Local method that needs a warm start; poor warm starts can be infeasible or converge to an equilibrium that is not the one you want.
- Parameters interact (penalty cap, trust-region floor, certificate tolerances); the defaults are not tuned for any one problem.
- Each iteration solves a joint SOCP whose size grows with the number of players and the horizon.
- Open-loop strategies only.

## API

The constructor that `DifferentialGamesBaseSolvers.jl` provides (the extension gives it a method once FALCON is loaded):

```@docs
falcon
```

The remaining types and functions (`FALCONSolver`, `TrajectoryScaling`, `ConvexifiedGame`) are documented in the FALCON repository.

## References

- [OutlandArya](@cite) B. Outland and V. Arya, *A Fast, Convergent Algorithm for Solving Non-convex Partially-Decoupled Generalized Nash Equilibrium Problems*, preprint.
