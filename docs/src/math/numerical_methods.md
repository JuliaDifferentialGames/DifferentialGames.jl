# Numerical Methods

Short descriptions of the methods implemented in `DifferentialGamesBaseSolvers.jl`, with pointers to the solver pages for details. The mathematical objects are defined in [Equilibrium Concepts](equilibrium_concepts.md) and [Optimality Conditions](optimality_conditions.md).

## Riccati recursion for LQ games — FNELQ

Backward pass computing the gains ``P_k`` and cost matrices ``Z_k`` by solving the stacked linear system ``S_kP_k=Y_k`` at each step; a forward pass produces the trajectory. Cost: one dense solve of size ``\sum_i m_i`` per step. Exact up to floating-point error when ``S_k`` is non-singular. [Page](../solvers/fnelq.md). The stochastic variant `SFNELQ` ([page](../solvers/sfnelq.md)) reuses this recursion for the supported noise model.

## Iterative LQ approximation — iLQGames

Repeat: roll out the current strategy; linearize the dynamics and quadraticize each player's cost around the trajectory; solve the resulting time-varying LQ game with FNELQ (with regularization ``\mu`` on ``S_k``); update the strategy with a feedforward line search; stop when the trajectory change is below `ε_conv`. Returns a local feedback-game approximation ([FridovichKeil2020](@cite)). [Page](../solvers/ilqgames.md).

## Augmented Lagrangian + Newton — ALGAMES

For constrained games: penalize constraints with weights ``\rho`` and multipliers ``\lambda``, solve the unconstrained penalized game's KKT system approximately with regularized Newton steps and an Armijo line search (inner loop), then update ``\lambda \leftarrow \max(0,\lambda+\rho g)`` and increase ``\rho`` (outer loop). Constraints are enforced at nodes ``1..N`` (optionally also at the terminal node). Based on [LeCleach2022](@cite). [Page](../solvers/algames.md).

## Other implemented methods

| Method | Solver | Page |
|--------|--------|------|
| Lexicographic iterated best response ([MillerMitra2022](@cite)) | `LIBR` | [page](../solvers/libr.md) |
| Distributed primal-dual (operator-splitting) algorithm for variational GNE of convex games ([YiPavel2017](@cite)) | `YiPavel` | [page](../solvers/yipavel.md) |
| Inverse LQ games: recover cost parameters from feedback gains | `InverseLQGames` | [page](../solvers/invlq.md) |
| Receding-horizon wrapper around any inner solver | `RecedingHorizonNash` | [page](../solvers/rhn.md) |
| Per-agent dynamic-tube MPC around a constrained game solver ([Lopez2019](@cite)) | `DynamicTubeMPC` | [page](../solvers/dtmpc.md) |
| Multi-agent actor-critic learning ([Lowe2017](@cite)), optionally guided by a local FNELQ solution | `MADDPGSolver` (package `DifferentialGamesMARL`) | [page](../marl/maddpg.md) |

Algorithmic details of these four are given on their pages; they have not been re-derived in this manual.
