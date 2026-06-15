# Inverse Games API

Complete reference for all inverse game types and functions.

## Problem Types

See [`InverseGameProblem`](../problem_types/inverse_game.md) for the full constructor reference.

## Player Knowledge

| Type | Description |
|------|-------------|
| [`KnownObjective`](../problem_types/inverse_game.md#Key-Types) | Player's cost is fully specified |
| [`UnknownObjective`](../problem_types/inverse_game.md#Key-Types) | Player's cost is to be inferred |

## Observation Models

| Type | Description |
|------|-------------|
| [`FullStateObservation`](../problem_types/inverse_game.md#Key-Types) | Full joint state observed |
| [`NoisyObservation`](../problem_types/inverse_game.md#Key-Types) | Noisy partial observation |

## Accessors

| Function | Description |
|----------|-------------|
| `unknown_players(prob)` | Indices of players with unknown objectives |
| `known_players(prob)` | Indices of players with known objectives |
| `n_unknown(prob)` | Number of unknown-objective players |
| `known_objective(prob, i)` | Known objective for player `i` |
| `as_forward_problem(prob, hyp)` | Build forward game from hypothesis |

## Solution and Observation Data

| Type / Function | Description |
|-----------------|-------------|
| `InverseGameSolution` | Holds inferred weights and history |
| `get_weights(sol)` | Final inferred weight vector |
| `get_weight_history(sol)` | Weight history across iterations |
| `ObservationData` | Stores observed trajectory data |
| `push_observation!(data, x, t)` | Append new observation |
