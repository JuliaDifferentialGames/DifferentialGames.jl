# Potential Games

Two problem types describe **finite state-based potential games** ([Marden2012](@cite)): a finite state space ``\mathcal X``, finite action sets ``\mathcal A_i``, utilities ``U_i(a,x)``, a (possibly stochastic) transition ``x'=T(a,x)``, and a potential function.

| Type | Condition on the potential |
|------|----------------------------|
| `StatePotentialGameProblem` | *exact*: the change in a player's utility from a unilateral deviation equals the change in the potential |
| `OrdinalPotentialGameProblem` | *ordinal*: only the sign is preserved (a recurrent state equilibrium exists by Lemma 3.1 of the reference) |

`LexicographicGameProblem` is itself an ordinal potential game; see [Lexicographic](lexicographic.md).

!!! note "Status"
    These types are **problem definitions only**: no solver in `DifferentialGamesBaseSolvers.jl` accepts them yet. They are provided so that learning dynamics and equilibrium computation for potential games can be implemented against the common interface (for example in an external solver package).

```@docs
StatePotentialGameProblem
OrdinalPotentialGameProblem
```

## Example: a two-state coordination game

Two players pick action 1 or 2 and are rewarded for matching; the state does not change. The utility of each player equals the potential, so the exact condition holds trivially.

```@example potential
using DifferentialGames

match(a, x) = a[1] == a[2] ? 1.0 : 0.0

game = StatePotentialGameProblem(
    [1, 2],                       # finite state space X
    [[1, 2], [1, 2]],             # action sets A_1, A_2
    Function[match, match],       # utilities U_i(a, x)
    (a, x) -> x,                  # transition x' = T(a, x)
    match,                        # potential φ(a, x)
)
game
```

```@example potential
# For this game the potential property can be checked by brute force over unilateral deviations:
ok = all(
    (game.utility_functions[i]((i == 1 ? a1p : a1, i == 2 ? a2p : a2), 1) -
     game.utility_functions[i]((a1, a2), 1)) ≈
    (game.potential((i == 1 ? a1p : a1, i == 2 ? a2p : a2), 1) - game.potential((a1, a2), 1))
    for i in 1:2, a1 in 1:2, a2 in 1:2, a1p in 1:2, a2p in 1:2)
ok
```
