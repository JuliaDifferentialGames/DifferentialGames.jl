# Information Structures

The information available to a player when it chooses ``u_k^i`` determines what a strategy can depend on, and therefore which equilibrium concept applies.

| Structure | Player ``i`` knows at step ``k`` | Strategy type | Built-in solvers |
|-----------|----------------------------------|---------------|------------------|
| **Open loop** | ``x_0`` and the game data only | `OpenLoopStrategy` | `ALGAMES`, `LIBR` (see their pages) |
| **Perfect state (feedback)** | the current joint state ``x_k`` | `FeedbackStrategy` | `FNELQ`, `SFNELQ`, `iLQGames`, `RecedingHorizonNash` (inner-solver dependent) |
| **Partial / noisy observation** | a function ``y_k = h(x_k) + \text{noise}`` of the state | belief-based | `PrivateObservation` (own signal) and `SharedObservation` (common signal) with `POGameProblem` ([Uncertainty](../problem_types/uncertainty.md)); no built-in forward solver is declared for them |
| **Private objectives** | its own cost; the others' costs may be unknown | — | inverse games ([InverseLQGames](../solvers/invlq.md)) infer them from observed play |
| **Asymmetric (Stackelberg)** | followers observe the leader's committed strategy | — | `AsymmetricInformation(leader_id)` is a placeholder; no solver yet |

## Consequences

- With **perfect state** information a feedback strategy reacts to disturbances and to other players' deviations; the resulting equilibrium is *time consistent* (subgame perfect) and is the solution of the Bellman recursion.
- With **open-loop** information the strategy is fixed in advance. Open-loop and feedback equilibria differ in general; an open-loop plan does not correct for the other players' unexpected behavior unless re-solved (as in receding-horizon use).
- `RecedingHorizonNash` re-solves a finite-horizon game from the current state at every step, giving closed-loop behavior from an open-loop or feedback inner solver.
- A `FeedbackStrategy` can be converted to an open-loop sequence along its nominal trajectory with `to_open_loop`.

## Where each choice appears in the API

`solver_capabilities` encodes this through `:FeedbackPolicies` and `:OpenLoopPolicies`; `GNEPSolution.strategy` holds the corresponding strategy object, and `is_feedback(sol)` / `is_open_loop_solution(sol)` test it.
