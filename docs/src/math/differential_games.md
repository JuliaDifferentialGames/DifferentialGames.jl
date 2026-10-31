# Differential Games

This page fixes the notation used throughout the manual. Other pages refer back to it rather than redefining symbols.

## General finite-horizon formulation

The setting is the standard one of [BasarOlsder1999](@cite).

There are ``P`` players, indexed ``i = 1,\dots,P``, and ``N`` time steps ``k = 0,\dots,N-1`` (the code indexes steps from 1). The joint state is ``x_k \in \mathbb{R}^n`` and player ``i`` chooses ``u_k^i \in \mathbb{R}^{m_i}``. Write ``u_k = (u_k^1,\dots,u_k^P)`` and ``u_k^{-i}`` for all controls except player ``i``'s.

```math
x_{k+1} = F_k(x_k, u_k^1, \dots, u_k^P), \qquad x_0 \text{ given},
```

```math
J_i\big(u^i, u^{-i}\big) = \phi_i(x_N) + \sum_{k=0}^{N-1} \ell_{i,k}\big(x_k, u_k^1,\dots,u_k^P\big).
```

Each player minimizes its own cost ``J_i``. Constraints come in two kinds:

- **Private** constraints ``h_i(x_k, u_k^i) \le 0``, which involve only player ``i``'s decisions (control bounds, state bounds);
- **Shared** constraints ``g(x_k, u_k) \le 0`` (or ``= 0``), which couple players (collision avoidance, formation).

!!! note "Cost scaling"
    The solvers' reported costs for LQ objectives include a factor ``\tfrac12``: ``J_i = \tfrac12[\sum_k x_k^\top Q_i x_k + u_k^{i\top} R_i u_k^i + x_N^\top Q_{f,i} x_N]``. See [FNELQ](../solvers/fnelq.md).

## Time discretization

Games are solved in discrete time. `LQGameProblem` and `LTVLQGameProblem` take discrete-time matrices directly. For `PDGNEProblem` you provide a continuous-time right-hand side ``\dot x^i = f^i(x^i, u^i)``; the solvers discretize it with step ``\Delta t`` (iLQGames: zero-order hold by default, `ZOHDiscretization()`; trajectory rollouts: RK4).

## How the problem types specialize the formulation

| Type | Dynamics | Costs | Constraints |
|------|----------|-------|-------------|
| [`LQGameProblem`](../problem_types/lq_game.md) | ``x_{k+1} = Ax_k + \sum_i B_i u_k^i`` on a shared state | quadratic | none |
| [`LTVLQGameProblem`](../problem_types/ltv_lq_game.md) | same, with ``A_k, B_{i,k}`` | quadratic, time varying | none |
| [`PDGNEProblem`](../problem_types/pdgnep.md) | **separable**: ``\dot x^i = f^i(x^i,u^i)``, ``x = (x^1,\dots,x^P)`` | any (functions of the **joint** state) | private + shared |
| `StochasticGameProblem` | LQ dynamics with noise (see [SFNELQ](../solvers/sfnelq.md)) | quadratic | none |
| `ConvexGameProblem`, `LexicographicGameProblem`, `InverseGameProblem`, `RecedingHorizonNashProblem` | see their solver pages | | |

In a `PDGNEProblem` the players are coupled **only** through costs and shared constraints. Which state a hand-written cost or constraint function receives depends on the solver (`ALGAMES`: joint; `iLQGames`: the player's own slice; see [Conventions](../ecosystem/conventions.md)); each player's own block starts at the offset recorded in `game.metadata.state_offsets`.

## Strategies

A *strategy* maps information to controls (see [Information Structures](information_structures.md)):

- open loop: ``u_k^i = \mu_k^i`` — a control sequence, independent of the realized state (`OpenLoopStrategy`);
- feedback: ``u_k^i = \gamma_k^i(x_k)`` — an affine law ``u_k^i = \hat u_k^i - P_k^i(x_k-\hat x_k) - \eta\,\alpha_k^i`` (`FeedbackStrategy`, with `get_gain` returning ``P_k^i``).
