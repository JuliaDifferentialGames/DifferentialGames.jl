# Notation

Symbols used throughout the manual. See [Differential Games](../math/differential_games.md) for the full formulation.

| Symbol | Meaning | In code |
|--------|---------|---------|
| ``P`` | number of players | `num_players(game)` |
| ``N``, ``\Delta t``, ``t_f`` | number of steps, step length, horizon (``t_f = N\Delta t``) | `n_steps(game)`, `game.time_horizon` |
| ``x_k \in \mathbb{R}^n`` | joint state at step ``k`` | `state_dim(game)`; columns of `sol.state_trajectory` |
| ``u_k^i \in \mathbb{R}^{m_i}`` | control of player ``i`` | `control_dim(game, i)`; `get_trajectory(sol, i).controls` |
| ``u_k^{-i}`` | all controls except player ``i``'s | — |
| ``F_k`` | discrete-time dynamics, ``x_{k+1}=F_k(x_k,u_k)`` | `game.dynamics` |
| ``\ell_{i,k}``, ``\phi_i`` | stage and terminal cost of player ``i`` | `PlayerObjective(i, stage, terminal)` |
| ``J_i`` | cost of player ``i`` | `get_cost(sol, i)` |
| ``A_k, B_{i,k}`` | LQ dynamics matrices | `LQGameProblem(A, B, …)` |
| ``Q_{i,k}, R_{i,k}, Q_{f,i}`` | LQ cost weights | `Q`, `R`, `Qf` |
| ``M_{i,k}, q_{i,k}, r_{i,k}`` | LQ cross and linear terms | `LQStageCost(Q, R, M, q, r)` |
| ``P_k^i`` | feedback gain of player ``i`` | `get_gain(strategy, i, k)` |
| ``\alpha_k^i`` | feedforward term, ``u=\hat u-P\,\delta x-\eta\alpha`` | `get_feedforward(strategy, i, k)` |
| ``Z_{i,k}`` | cost-to-go matrix, ``V_{i,k}=\tfrac12x^\top Z_{i,k}x`` | `sol.solver_info[:cost_to_go_matrices]` |
| ``S_k`` | stacked gain system of the Riccati step | `sol.solver_info[:rcond_history]` (its conditioning) |
| ``h_i``, ``g`` | private and shared constraints (``\le 0``) | `control_bounds`, `SharedInequality`, … |
| ``\lambda``, ``\rho`` | multipliers and penalty (ALGAMES) | `sol.solver_info[:dual_variables]` |
| ``\varepsilon_{\mathrm{conv}}`` | iLQGames trajectory-change tolerance | `iLQGames(; ε_conv)` |
| ``\mu`` | regularization of ``S_k`` | `iLQGames(; μ_init)`, `FNELQ(; regularization)` |
