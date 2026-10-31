# Documentation Audit — Phase 1 (PR 1: documentation correctness)

Scope: the umbrella documentation in `docs/src`, `README.md`, and the install path.
Method: every ```` ```julia ```` block was extracted and executed against the current `main` branches of
`DifferentialGamesBase.jl` and `DifferentialGamesBaseSolvers.jl` (Julia 1.12.3); installation was tested in a clean
depot; solver capability lists and constructors were read from the source. Pages that are fixed now use
` ```@example ` blocks, so the strict Documenter build (`warnonly = false`) executes them on every build.

Status vocabulary:

- **Confirmed** — observed by running code.
- **Documented** — stated by a docstring or reference page; not independently re-verified here.
- **Needs validation** — a claim or behavior that could not be settled in this pass.

## 1. Findings fixed in this PR

| # | Location | Problem (Confirmed) | Resolution |
|---|----------|---------------------|------------|
| 1 | `README.md`, `index.md`, `first_example.md`, `installation.md`, `lq_games.md` | `LQGameProblem(0.95*I(n), …)` throws `MethodError`: the constructor needs concrete `Matrix{T}` / `Vector{Matrix{T}}`; `I(n)` and `Diagonal` are rejected. The README quick start did not run. | Examples use `Matrix(...)`; the requirement is documented on the problem-type pages. |
| 2 | `README.md`, `pdgnep.md`, `nonlinear_games.md` | `iLQGames` on a game with a `SharedInequality` throws `AssertionError: iLQGames handles only unconstrained games`. | README and `pdgnep.md` use `ALGAMES` for constraints; nonlinear tutorial is unconstrained. |
| 3 | `api/solver_interface.md` | Documented `AbstractGameSolver` and custom `solve` methods; the real interface is `GameSolver` + `_solve(game, solver, warmstart, verbose)`. | Rewritten; example is executed. Found a pitfall: an unannotated `warmstart` argument makes a custom `_solve` **ambiguous** with the base fallback — documented. |
| 4 | `constrained_games.md` | (a) `ALGAMES(; verbose=true)`, `max_outer`, `max_inner`, `ρ_scale`, `ε_primal`, `ε_dual` do not exist (real: `outer_iter`, `inner_iter`, `ρ_increase`, `tol_opt`, `tol_dyn`, `tol_con`). (b) Terminal costs for players 2–3 indexed `x[1:2]` of the **joint** state, i.e. penalised player 1's position (players 2 and 3 did not move). (c) The constraint check concatenated only two player states and tested only the last node; trajectories already hold the joint state. (d) The initial positions violated the 1.5 m separation. | Rewritten as a two-vehicle example with per-player offsets, a feasible start, and a check over **all nodes**; assertions on the constraint and control bounds. States that the collision constraint is inactive in the reported solution. |
| 5 | `lq_games.md` | `sol.solver_info[:feedback_gains]` does not exist (`KeyError`); the LTV example had syntax errors and was commented out; "FNELQ on a `PDGNEProblem`" asserts `FNELQ requires LinearDynamics`. | Gains via `get_gain(strategy, i, k)`; closed-loop rollout check; executable LTV example; LQR-limit check against the Riccati recursion; `PDGNEProblem` limitation documented. |
| 6 | `installation.md` | `Pkg.add(url=…/DifferentialGames.jl)` fails in a clean depot with `Unsatisfiable requirements detected for package DifferentialGamesBaseSolvers`, because Base/BaseSolvers are unregistered. Julia compat of `1.10` contradicts `DifferentialGamesBaseSolvers` (`julia = "1.11"`). | Install sequence adding all three by URL in one `Pkg.add` (tested in a clean depot: loads successfully). Requirements state Julia ≥ 1.11. `Project.toml` compat set to `1.11` and CI matrix drops 1.10. |
| 7 | `ilqgames.md`, `algames.md`, `libr.md`, `yipavel.md`, `invlq.md`, `solver_options.md` | `verbose` listed as a solver constructor option (it is a `solve` keyword; the constructors have no such field); stale option names; stale capability symbols (`:lq_game`, `:feedback_nash`, …). | Corrected to the real option lists and `solver_capabilities` values. |
| 8 | `ilqgames.md`, `lq_game.md` | "converges in 1 iteration on LQ games". **Observed: 15 iterations** on the README LQ example (cost within ~4·10⁻⁴ relative of `FNELQ`). | Claim corrected. |
| 9 | `algames.md` | "converges to a Normalized Nash Equilibrium" stated unconditionally. | Replaced by a qualified statement (see §3). |
| 10 | `solutions.md` | `get_costs` documented as `Dict`; actual `Vector`. `solver_info` keys (`:feedback_gains`, `:Δ_history`, `:constraint_violation`, …) do not exist. `rollout_strategy(game, strat, x0)` has a different signature. | Corrected from observed `solver_info` keys and method tables. |
| 11 | `problem_interface.md` | `NonlinearTerminalCost(x -> …)` fails (terminal costs take `(x, p)`); `collision_avoidance` keywords (`state_offset_i`, `d_safe`, `state_dim`) do not exist (real: `i_offset`, `j_offset`, `pos_dim`, `d_min`); `player_slice(1,4,8)` has a different signature. | Corrected / marked unverified. |
| 12 | `ltv_lq_game.md` | Argument table said `B_seq[k][i]`; code uses `B_seq[i][k]`; example mixed shapes. | Single, executed example. |
| 13 | `make.jl`, CI | `warnonly = true` hid errors. A strict build immediately surfaced: `@docs` entries without docstrings (`GameSolver`), duplicate `WarmstartData`, an unresolved `@ref` to `CANONICAL_CAPABILITIES`/`InverseLQGameSolution`, and `unknown_players`. | `warnonly = false`, `doctest = true`; docs now have an explicit `docs/Project.toml` (`[sources]`) instead of `LOAD_PATH` manipulation and ad hoc `Pkg.add`. |

## 2. Library issues found in sibling packages

Status after the follow-up pass. Changes were made in the sibling working trees (not committed); both packages'
test suites pass (Base 1171 pass / 1 broken; BaseSolvers 572+ pass including the new `test/regression_tests.jl`).

| # | Package | Issue | Status |
|---|---------|-------|--------|
| A | BaseSolvers | Leftover debug code in `iLQGames` (`apply_strategy_debug`, `@info "lq_sol strategy feedforward"` inside the μ-escalation loop). The `apply_strategy_debug` block only ran with `verbose=true` — the earlier claim that it ran with `verbose=false` was wrong. | Debug code **removed** |
| B | BaseSolvers | `iLQGames` did not declare `:SeparableDynamics`, so every `PDGNEProblem` warned. | **Fixed** (capability added; tested) |
| C | BaseSolvers + Base | `iLQGames` did not converge on an exactly linear-quadratic `PDGNEProblem` (first full step *increased* cost 90 → 1821; exact per-player cost 7.9835). **Root cause: a sign error in `FNELQ`'s affine (linear-term) recursion** — `α` was computed as `-S⁻¹(Bᵀζ+r)` but applied as `u = -Px - α` in FNELQ's own forward pass and in the `ζ` recursion; `apply_strategy` compensated with `+ηα` while its docstrings state `-ηα`. | **Fixed**: FNELQ now computes `α = S⁻¹(Bᵀζ+r)`; `apply_strategy` implements the documented `ûᵢ - Pᵢδx - ηαᵢ`. Verified on scalar problems with known solutions and on the LQ `PDGNEProblem` (costs → 7.98); unicycle example converges in 4 iterations instead of 40. Regression tests added. |
| D | BaseSolvers | **Fixed (opt-in).** `ALGAMES` reported `converged = true`, `con_vio = 0` while a constraint was violated by 1.04 at the last node. **Cause**: constraints (and `con_vio`) are evaluated on `(x_k, u_k)` for `k = 1..N`; the terminal state `x_{N+1}` is never constrained. | New option `ALGAMES(; terminal_constraints=true)` enforces state-only constraint rows at `x(N+1)` (control-dependent rows such as bounds cannot apply to a state without a control and are excluded there); default unchanged. Regression test added. With default options the same problem can also end with `converged = false` and small `con_vio` (stationarity residual stays large); that behavior is unexplained. |
| E | Base | Claimed `show` of a `PDGNEProblem` solution throws. | **Retracted** — could not be reproduced; `show` works. |
| F | Base | Docstring for `unknown_players(::InverseGameProblem)` was attached to `n_players`. | **Fixed** (moved) |
| G | Base | `GameSolver` had no docstring. | **Fixed** (docstring added, including the `warmstart` annotation pitfall) |
| H | Base | `LQGameProblem` accepted only concrete `Matrix{T}`; `I(n)` gave a `MethodError`. | **Fixed**: converting method accepts any `AbstractMatrix`/`AbstractVector` (→ `Float64`); strict method now `where {T<:Real}` to avoid ambiguity.. Tested. |
| I | BaseSolvers | `animate_solution`/`save_animation` were only error stubs (the Plots extension never implemented them; the docstring described shapes that did not exist). | **Fixed**: general implementation in the Plots extension (any `GNEPSolution`, `xy_indices`, `subsample`); the stub no longer collides with the extension method. Verified manually (GIF written); no automated test because Plots is a weak dependency. |
| J | Both | Uncommitted work in `DifferentialGamesBaseSolvers` (`SFNELQ`, capability-vocabulary test) pre-dates this audit. | `SFNELQ` now has a documentation page |

### 2b. Cross-package findings from documenting MARL, Benchmarks and the remaining solvers

| # | Where | Finding | Status |
|---|-------|---------|--------|
| K | Base `expand` vs `ALGAMES` | **Cost-function argument convention differs between solvers.** For a `NonlinearStageCost`/`NonlinearTerminalCost` in a `PDGNEProblem`, `ALGAMES` passes the **joint** state and **joint** control; `iLQGames`/`expand` pass **player i's own** state and control slices (checked by instrumenting the functions; a joint-indexing cost raises `BoundsError` under iLQGames). Consequences: costs are not portable between solvers; `0.01*sum(abs2,u)` charged each player for all players' effort under ALGAMES (the earlier tutorial/README examples did this; fixed to index own controls); MARL's built-in tasks (written for joint state) make `build_guidance` fail. The comment in `quadraticize_costs` ("full joint state for nonlinear costs") disagrees with its behavior. | **Documented** ([Conventions](https://JuliaDifferentialGames.github.io/DifferentialGames.jl/dev/ecosystem/conventions/)) with a portable idiom (verified with both solvers); **not changed** — choosing one convention is a cross-package design decision |
| L | Base `determinize` | The determinized game's dynamics is a discrete-time map wrapped in `CoupledNonlinearDynamics`, which solvers integrate as continuous time: `solve(determinize(sg), iLQGames())` fails (`InexactError: Int64(0.05)`), contradicting the docstring. | Documented (limitation); needs a discrete-time nonlinear dynamics type |
| M | Solvers `DynamicTubeMPC` + `ALGAMES` | `warm_start = true` failed: the shift assumed per-player private trajectories while ALGAMES stores joint states. `_shift_trajectories` (RHN) now handles joint-state trajectories (regression test added), but DTMPC's own `_fix_dtmpc_warmstart_tail` still assumes private augmented states, so use `warm_start = false` with ALGAMES. | Partly fixed; documented |
| N | Solvers `DTMPC.md` | The in-repo example uses an outdated constructor (`α₀`, …) and fails (`MethodError`); the source header says the old API was removed. | Documented with a corrected, locally run example (not executed in the manual) |
| O | Solvers `LIBR` | The lexicographic priority is enforced by a quadratic penalty; with default `col_penalty = 1e3` a head-on test ended with separation 0.17 (collision distance 0.3); `1e9` gave 0.29. | Documented; tutorial uses `1e9` and asserts the separation |
| P | MARL | `build_guidance`/`gps_builder` fail on `cooperative_navigation` (cause K) and degrade to unguided training (`"builder declined"`); work on LQ-cost games (verified). The training loop and GPS have **no tests**. | Documented |
| Q | MARL/Base | `GNEPSolution`'s docstring says `state_trajectory` is `nothing` for PD-GNEPs (private states); `iLQGames`/`ALGAMES` set it and store joint states in each `Trajectory`, `LIBR` follows the docstring. | Documented in Conventions |
| R | Benchmarks | Independently found and reproduced the `FNELQ` affine-term sign error (see its README); the LQ formation case works around it by using a zero offset target. The fix in the Solvers tree (item C) removes the need for the workaround; the case has not been re-run with a nonzero offset. | Cross-reference; not re-verified |
| S | Benchmarks README | States `iLQGames` "will error" on the pursuit–evasion cases because it does not declare `:SeparableDynamics`; that capability is now declared (item B), though iLQGames still cannot handle constraints. | Cross-reference |

## 3. Claims that still need validation

| Claim | Where | Why not settled |
|-------|-------|-----------------|
| ALGAMES returns a normalized / variational GNE | `algames.md` | Paper-level claim; not tested on this implementation. Page now says so. |
| Reported costs include a factor ½ for **nonlinear** costs | `solutions.md` | Confirmed for LQ costs only (LQR comparison: `J = ½ x₀ᵀP₀x₀`). |
| `FNELQ` Riccati recursion as printed on `fnelq.md` | `fnelq.md` | The older notation on that page was not re-derived. The recursion on the new [Optimality Conditions](math/optimality_conditions.md) page matches the code; output is verified against single-player Riccati and closed-loop rollout. |
| `FNELQ` uniqueness / well-posedness conditions | `fnelq.md` | Only `rcond` diagnostics documented. |
| iLQGames → ALGAMES warm start benefit | `algames.md` | The combination runs; improvement not benchmarked. |
| Complexity statements | `fnelq.md` | Not measured. |
| `LIBR`, `YiPavel`, `RecedingHorizonNash`, `InverseLQGames`, `SFNELQ` page content (algorithm descriptions, option tables, examples) | `solvers/*.md`, `inverse_game.md` | Option names were checked against struct fields; the examples and mathematical statements were **not** re-executed. They depend on `ConvexGameProblem`, `LexicographicGameProblem`, `InverseGameProblem`, `RecedingHorizonNashProblem` fixtures. |
| `basics/overview.md`, `cost term DSL` in `problem_interface.md` | – | Not executed. |
| Stochastic FNELQ (`SFNELQ`) assumptions and limitations | – | No page yet (Phase 3). |

## 4. Page status after this PR

| Page | Examples executed by the build | Notes |
|------|--------------------------------|-------|
| `index.md`, `getting_started/first_example.md`, `getting_started/installation.md` | yes | |
| `tutorials/lq_games.md`, `nonlinear_games.md`, `constrained_games.md` | yes | |
| `problem_types/pdgnep.md`, `ltv_lq_game.md` | yes | |
| `api/solver_interface.md` | yes (custom solver) | |
| `problem_types/lq_game.md`, `solvers/fnelq.md`, `ilqgames.md`, `algames.md` | no | Fragments; corrected against source and run examples. |
| `solvers/libr.md`, `rhn.md`, `yipavel.md`, `invlq.md`, `problem_types/inverse_game.md` | no | See §3. |
| `basics/*` | no | Corrected where checked (§1 items 10–11). |

## 5. Added in the follow-up pass (PR 2 content)

- `math/`: Differential Games (shared notation), Equilibrium Concepts, Information Structures, Optimality Conditions, Numerical Methods.
- `solvers/sfnelq.md` (new, executed example) and `basics/solver_page_template.md`; FNELQ, iLQGames, ALGAMES pages gained *Assumptions* and *Known limitations*.
- `basics/solver_selection.md` populated from `solver_capabilities`.

## 6. Added with the ecosystem pass

- `ecosystem/`: overview (packages, dependency graph, routes), conventions across packages, Benchmarks.
- `problem_types/`: overview and hierarchy, Lexicographic, Convex, Potential, Stochastic/Robust/Partially observable (executed examples where possible).
- `solvers/dtmpc.md`; examples added to the RHN page.
- `marl/`: overview, environments, MADDPG, guided policy search (code verified, not executed in the build).

## 6b. Added with the benchmark-suite pass

- `benchmarks/`: overview and catalogue, Hallway passing, Racing (Austin), Multi-agent particle tasks (Lowe et al.). Code samples run against the current sources; not executed in the build.
- `DifferentialGamesBenchmarks.jl` (uncommitted): `HallwayPassing.Nominal_v1` and `Racing.AustinNominal_v1` ported from the FALCON suite, with warm starts, evaluators, visualizers (walls are drawn through a new optional `walls` field in the visual model), `data/f1_austin_segment.csv` and `test/falcon_cases_test.jl`.
- `DifferentialGamesMARL.jl` (uncommitted): `keep_away` and `physical_deception` (`src/target_adversary.jl`) with tests. Cooperative communication and covert communication are **not** implemented: they need a message channel `MADDPGEnv` does not have.

Findings:

- **`:nash_residual` is misnamed.** `game_quality_nash_residual` compares recomputed with *reported* cost; it does not test optimality and returned 0 for a racing solution whose own optimality residual was of order 10³. The ecosystem and benchmark pages previously presented it as a Nash residual; they now describe it as a cost-consistency check. A genuine equilibrium test (best response against the others' fixed trajectories) does not exist yet.
- **Integration step matters for stiff dynamics.** With the Pacejka tyre model one RK4 step per 0.1 s gives a different trajectory from five substeps (cars 0.055 m apart instead of 0.6 m for the same controls); the racing benchmark therefore uses `dt = 0.05`. The Base solvers have no map-style (pre-integrated, sub-stepped) nonlinear dynamics type, which is what the FALCON script uses (same gap as the `determinize` limitation in 2b).
- **`ALGAMES` does not solve the racing game.** The line search returns `α = 0` in the first inner iteration of every outer iteration; the returned trajectory is the warm-start rollout. Regularization and penalty changes do not help; without a warm start the cars do not move. Not changed: the point of the benchmark is that it is hard.
- **`ALGAMES` on the hallway** needs the warm start (the straight-line initialization crosses the walls) and about ten minutes; see the hallway page for the result.
- **`keep_away` / `physical_deception` via a model-based solver solve the complete-information game.** The adversary's ignorance of the target lives only in `obs_fn`, so `ALGAMES` lets the adversary reach the target; the deception studied in the paper arises only in the learner. Documented as a warning on the page.
- The paper's contact pushing (keep-away) is replaced by agent–adversary collision constraints, because contact forces would couple the agents' dynamics and break `SeparableDynamics`.

## 6c. DGSQP

- `DifferentialGamesBaseSolvers.jl` (uncommitted): `DGSQP` solver (`src/solvers/DGSQP`), new dependency Clarabel, tests (`dgsqp_tests.jl`), `paper_study.jl`, `.github/workflows/DGSQP.yml`, and the example `curved_track.jl`. Docs page `solvers/dgsqp.md`.
- Verified: derivatives against finite differences; analytic 2-variable game; agreement with `ALGAMES` on an LQ game; best-response certificates; the curved-track Monte Carlo matches Table I of the paper (success rates and QPs per solve; see the page).
- Not replicated: the ALGAMES column of Table I (different discretization and previous-input terms), the 1000-trial racetrack study, the merit-function ablation, BFGS Hessians, IBR/PATH.
- The monotone line search with the reference's `beta = 0.01` stalls on the lane-swap game; the paper's non-monotone search is needed (documented).

## 6d. DGSQP on the Austin hairpin

- Added the racing-paper variant (`dgsqp_racing_options`: decaying regularization, d/m-steps, merit memory), `SeparableRolloutGame` (exact structured derivatives, equal to whole-rollout AD to 1e-15, 0.14 s per evaluation at 420 variables versus minutes), `examples/austin_hairpin.jl`, tests and docs.
- Result: converges at N = 25 and N = 40; does **not** converge at N = 55 and N = 70 (the FALCON figure). A single-agent warm start did not help (infeasible first QP; the reference's elastic mode is not ported).
- Not run: FALCON itself, and DGSQP on the `Racing.AustinNominal_v1` `GameProblem`.

## 6e. DG-SQP vs FALCON on the Austin benchmark

- Reproduced the FALCON paper's comparison (N = 25, 50 initial conditions, single-agent FALCON warm starts): Julia DG-SQP 33/50 converged (66 %, paper's Python: 72 %), 394 ± 56 iterations (paper 423 ± 138); FALCON at the paper commit 50/50 with 0.873 ± 0.407 s and 85.9 ± 43.8 iterations, matching the paper.
- **FALCON parameters, not a regression:** the current working tree (`2ba0f55` plus uncommitted edits) added a mandatory ζ-certificate (`ζ_cert_*` defaults) that rejects most SOCP solves of the racing game: 6/50 converged with the paper's settings. With `ζ_cert_abs_tol=1e-2, ζ_cert_rel_tol=1e-1, ζ_cert_polish_ΔJ_tol=1e2`: 49/50 (0.733 ± 0.217 s, 91.6 ± 23.9 iterations). The paper commit `6db00e7`: 50/50. The `FALCON_CERT=loose` switch of `falcon_export.jl` selects the loosened certificate.
- DG-SQP failures: 14 diverged (stat > 1e10 after a relaxed step), 3 infeasible first QP (no elastic mode).
- Files: `examples/austin_benchmark.jl`, `falcon_export.jl`, `data/` (ICs, Python results, Julia and FALCON results), `test/austin_benchmark_tests.jl` (skipped when DifferentialGamesBenchmarks is not installed).

## 6f. Tuning, FALCON examples in Benchmarks, release preparation

- FALCON failure diagnosis (working tree, paper settings, 6/50): (1) the mandatory ζ certificate's defaults reject most racing SOCP solves; (2) `ω_max = 1e3` caps the penalty before χ reaches 1e-4 (IC 38); (3) with `ω_max = 1e4` the trust region collapses at `r_min = 1e-3` (IC 40). Tuned settings (ω_max 1e4, r_min 1e-5, certificate 2e-3/1e-2/1): 50/50, 0.777 ± 0.180 s, 94.6 ± 22.6 iterations. Tuned on these 50 initial conditions only.
- DGSQP tuning: `dgsqp_tuned_options()` (reg 1e3, decay 0.9, delta decay 0.95, elastic QP, divergence recovery): 50/50, 362 ± 59 iterations, 20.6 ± 5.1 s (reference settings: 33/50). `elastic_mode` and `divergence_recovery` are extensions of the reference; the reference defines an elastic QP but never calls it.
- Benchmarks: `Racing.AustinHairpin_v1`, `HallwayPassing.Communication_v1`, `austin_benchmark_initial_conditions()`, `racing_game_from_ic`. The spacecraft examples of the paper (lady-bandit-guard, inspection denial) are not in the FALCON examples directory and were not ported.
- FALCON repo: README, CITATION.cff, `[sources]` for DifferentialGamesBase, the tuned racing parameters listed in the README (585 tests pass in a scratch copy). **Not done / for the owner:** `examples/f1_austin_racing copy.jl` is a stale duplicate (left in place), the test files `include` source files into `Main` and produce method-overwritten warnings, no documentation site, nothing committed in any repository.

## 7. Remaining work

- Bring `LIBR`, `YiPavel`, `InverseLQGames`, `RecedingHorizonNash` to the ten-section template and execute their examples.
- PR 3: flagship tutorials via Literate.jl, extension guide, DocumenterInterLinks and an external-solver index, plots.
- `checkdocs = :exports` after an API-docstring completeness audit across Base/BaseSolvers.
- ALGAMES' default `tol_opt = 1e-4` is often unattainable at an active control bound: stationarity stalls near 2·10⁻³ (non-smooth active set), the line search then fails (`α ≈ 0`) and further penalty growth destabilizes later iterations (opt jumps to O(10)). This is numerical behavior of the method, not a logic error; use `tol_opt ≈ 1e-2` or stop on `con_vio`. Not changed.
