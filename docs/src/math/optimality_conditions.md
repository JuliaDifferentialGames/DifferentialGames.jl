# Optimality Conditions

## Feedback Nash: coupled Bellman recursion

See [BasarOlsder1999](@cite), Ch. 6, for the discrete-time derivation.

Let ``V_{i,k}(x)`` be player ``i``'s cost-to-go from state ``x`` at step ``k``, with ``V_{i,N} = \phi_i``. A feedback Nash equilibrium ``\gamma^*`` satisfies, for all ``i, k, x``,

```math
V_{i,k}(x) = \min_{u^i}\Big\{ \ell_{i,k}\big(x, u^i, \gamma_k^{-i*}(x)\big) + V_{i,k+1}\big(F_k(x, u^i, \gamma_k^{-i*}(x))\big) \Big\},
```

and ``\gamma_k^{i*}(x)`` attains the minimum.

### LQ case: coupled Riccati recursion (as implemented in FNELQ)

For ``x_{k+1} = A_k x_k + \sum_j B_{j,k} u_k^j`` with stage cost ``\tfrac12(x^\top Q_{i,k}x + u^{i\top} R_{i,k}u^i)`` (the implementation also supports cross terms ``M_{i,k}`` and linear terms ``q_{i,k}, r_{i,k}``), the cost-to-go is ``V_{i,k}(x) = \tfrac12 x^\top Z_{i,k} x`` with ``Z_{i,N} = Q_{f,i}``. At each step solve the linear system for the stacked gain ``P_k = [P_k^1; \dots; P_k^P]``:

```math
S_k P_k = Y_k,\qquad
S_k^{ii} = R_{i,k} + B_{i,k}^\top Z_{i,k+1} B_{i,k},\quad
S_k^{ij} = B_{i,k}^\top Z_{i,k+1} B_{j,k}\ (j\neq i),\quad
Y_k^{i} = B_{i,k}^\top Z_{i,k+1} A_k ,
```

then, with ``F_k = A_k - \sum_j B_{j,k}P_k^j``,

```math
Z_{i,k} = F_k^\top Z_{i,k+1} F_k + Q_{i,k} + P_k^{i\top} R_{i,k} P_k^i .
```

The equilibrium strategy is ``u_k^i = -P_k^i x_k``. When the linear terms are present the cost-to-go gains a linear part ``\zeta_{i,k}^\top x`` and the law becomes ``u_k^i = -P_k^i x_k - \alpha_k^i`` with ``\alpha_k = S_k^{-1}\big(B_k^\top\zeta_{k+1} + r_k\big)`` (blockwise per player). Solvability requires ``S_k`` to be non-singular; FNELQ reports `rcond(S_k)` in `sol.solver_info[:rcond_history]`.

This recursion was checked numerically in this documentation: for a single player it reproduces the Riccati LQR cost, and rolling out ``u = -Px`` reproduces the stored trajectory (see the [LQ tutorial](../tutorials/lq_games.md)).

## Open-loop Nash: first-order conditions

For differentiable problems an OLNE satisfies, for each player, the necessary conditions of an optimal control problem with the others' controls held fixed. With costates ``\lambda_{i,k}``:

```math
\lambda_{i,k} = \nabla_x \ell_{i,k} + \big(\partial_x F_k\big)^\top \lambda_{i,k+1}, \qquad \lambda_{i,N} = \nabla\phi_i(x_N),
```

```math
0 = \nabla_{u^i} \ell_{i,k} + \big(\partial_{u^i} F_k\big)^\top \lambda_{i,k+1}.
```

Stacking these for all players with the dynamics gives one coupled nonlinear system (the "game KKT system") in the states, controls and costates. ALGAMES solves this system by Newton's method (with regularization) on an augmented Lagrangian.

## Generalized Nash: coupled KKT with shared constraints

With constraints ``h_i \le 0`` (private) and ``g \le 0`` (shared) and multipliers ``\mu_i \ge 0``, ``\lambda_i \ge 0``, the KKT conditions of player ``i``'s problem are

```math
\nabla_{u^i} J_i + \nabla_{u^i} h_i^\top \mu_i + \nabla_{u^i} g^\top \lambda_i = 0,\quad
\mu_i \perp -h_i,\quad \lambda_i \perp -g .
```

A **variational** GNE additionally has ``\lambda_i = \lambda`` for all ``i``. ALGAMES applies the same dual-ascent update to the shared-constraint multipliers of all players (augmented Lagrangian, penalty ``\rho``), which is the mechanism by which such a point is sought.

## What a solver reports

Residuals are defined with respect to the system a solver actually solves:

| Solver | Residuals available in `solver_info` |
|--------|--------------------------------------|
| FNELQ | `:min_rcond`, `:rcond_history` (conditioning of ``S_k``) |
| iLQGames | `:trajectory_change_history`, `:cost_history` |
| ALGAMES | `:opt_vio` (stationarity), `:dyn_vio` (dynamics), `:con_vio` (constraints) per outer iteration |

None of these is a certificate that the equilibrium definition holds globally; they are local residuals.
