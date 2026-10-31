# Solver Selection Guide

This page is generated from the solvers' `solver_capabilities` and the `_solve` method signatures in `DifferentialGamesBaseSolvers.jl`, not inferred from algorithm names. Cells marked *unverified* are not yet confirmed by a test in this documentation.

## Decision workflow

1. **Is the game linear-quadratic on a shared state** (`LQGameProblem` / `LTVLQGameProblem`)? → [`FNELQ`](../solvers/fnelq.md) for a feedback Nash equilibrium (exact).
2. **Are there stochastic LQ dynamics?** → `SFNELQ` on a `StochasticGameProblem`.
3. **Nonlinear dynamics or costs, no constraints, feedback strategies wanted?** → [`iLQGames`](../solvers/ilqgames.md).
4. **Control bounds, state bounds or shared (coupling) constraints?** → [`ALGAMES`](../solvers/algames.md) (open-loop strategies).
5. **Lexicographic / prioritised objectives?** → [`LIBR`](../solvers/libr.md) on a `LexicographicGameProblem`.
6. **Convex games with shared constraints, generalized or variational Nash?** → [`YiPavel`](../solvers/yipavel.md) on a `ConvexGameProblem`.
7. **Recovering cost parameters from observed play?** → [`InverseLQGames`](../solvers/invlq.md) on an `InverseGameProblem`.
8. **Repeated solves along a moving horizon?** → [`RecedingHorizonNash`](../solvers/rhn.md) wrapping any inner solver.
9. **Bounded disturbances on a constrained receding-horizon game?** → [`DynamicTubeMPC`](../solvers/dtmpc.md).
10. **Only a simulator, or very non-smooth costs?** → the learning route: [MADDPG](../marl/maddpg.md) in `DifferentialGamesMARL`, optionally [guided](../marl/gps.md) by `FNELQ`.

## Capability matrix

| Solver | Problem type | Declared capabilities (`solver_capabilities`) | Strategy / solution | Constraints |
|--------|--------------|-----------------------------------------------|---------------------|-------------|
| `FNELQ` | `GameProblem` with `LinearDynamics` and LQ costs | `:LQGame :FeedbackPolicies :UnconstrainedGame :DiscreteTime` | Feedback Nash (`FeedbackStrategy`) | None |
| `SFNELQ` | `StochasticGameProblem` | `:LQGame :Stochastic :FeedbackPolicies :DiscreteTime` | Feedback Nash under its noise model | Not declared (*unverified*) |
| `iLQGames` | `GameProblem` | `:NonlinearGame :LQGame :UnconstrainedGame :SeparableDynamics :DiscreteTime` | Local feedback-game approximation (`FeedbackStrategy`) | **None** (asserts) |
| `ALGAMES` | `GameProblem` (`PDGNEProblem`) | `:LQGame :NonlinearGame :GNEP :ConstrainedGame :UnconstrainedGame :DiscreteTime :SeparableDynamics :SharedConstraints` | Open-loop equilibrium approximation (`OpenLoopStrategy`) | Private and shared |
| `DGSQP` | `GameProblem` (`PDGNEProblem`, `LQGameProblem`) or `OpenLoopGame` | `:LQGame :NonlinearGame :GNEP :ConstrainedGame :UnconstrainedGame :DiscreteTime :SeparableDynamics :SharedConstraints` | Open-loop equilibrium (`OpenLoopStrategy`), one QP per iteration, non-monotone line search | Private and shared |
| [`FALCON`](../solvers/falcon.md) (soft dependency, `using FALCON`) | `PDGNEProblem` | `:GNEP :NonlinearGame :SeparableDynamics :SharedConstraints :ConstrainedGame :PD_GNEP :DiscreteTime` | Open-loop equilibrium, one joint SOCP per iteration (sequential convexification, augmented Lagrangian, trust region) | Private and shared (coupling) |
| `LIBR` | `LexicographicGameProblem` | `:LexicographicGame :OpenLoopPolicies :DiscreteTime :SeparableDynamics` | Open-loop; priority-ordered | Collision penalty/slack options (*unverified*) |
| `YiPavel` | `ConvexGameProblem` | `:ConvexGame :GeneralizedNash :VariationalNash :SharedConstraints :SeparableDynamics :DiscreteTime` | Generalized / variational Nash (per declared capabilities) | Shared |
| `InverseLQGames` | `InverseGameProblem` | `:InverseLQGame :LQGame :FeedbackPolicies :InfiniteHorizon` | Recovered cost parameters (not a Nash solution) | N/A |
| `RecedingHorizonNash` | `RecedingHorizonNashProblem` | `:RecedingHorizon :FeedbackPolicies` | Sequence of inner solutions | Inherited from inner solver |
| `DynamicTubeMPC` | `DynamicTubeMPCProblem` (wraps a `PDGNEProblem`) | see its page | Closed-loop plan with tube widths | Coupled constraints tightened by tubes; inner solver dependent |
| `MADDPGSolver` (MARL package; **not** a `GameSolver`) | any `GameProblem` through `MADDPGEnv` | — | Neural-network policies; `train!`, no `GNEPSolution` | Reward shaping (barrier/penalty) and optional projection |

The `required_capabilities(game)` function reports what a `GameProblem` needs; `solve` warns on a mismatch .

## Rules of thumb

- Declared capabilities are a necessary hint, not a guarantee: always read the solver's *Known limitations*.
- A `converged = true` flag reports the solver's own stopping test. It is not an equilibrium certificate.
- Different local solvers can return different local solutions of the same nonlinear problem.
