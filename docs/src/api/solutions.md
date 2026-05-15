# Solutions API

## Solution Types

```@docs
GNEPSolution
```

## Accessor Functions

```@docs
get_trajectory
get_cost
first_step_state
n_players
n_steps
WarmstartData
```

## Solution Layout (PDGNEProblem)

For `PDGNEProblem`, each player has a private `Trajectory{T}` with fields:
- `states::Matrix{T}` — `(nᵢ × N+1)`; `states[:, k]` is the state at step `k`
- `controls::Matrix{T}` — `(mᵢ × N)`; `controls[:, k]` is the control at step `k`
- `times::Vector{T}` — time grid of length `N+1`
- `cost::T` — total accumulated cost for this player

```julia
sol = solve(game, iLQGames())

traj1 = get_trajectory(sol, 1)       # player 1's Trajectory
x1_k  = traj1.states[:, k]           # player 1's state at step k
u1_k  = traj1.controls[:, k]         # player 1's control at step k
J2    = get_cost(sol, 2)              # player 2's total cost
```

## Solution Layout (LQGameProblem)

For shared-state games, the `state_trajectory` field holds the joint state:

```julia
sol = solve(game, FNELQ())

x_k  = sol.state_trajectory[:, k]         # joint state at step k
u1_k = sol.trajectories[1].controls[:, k] # player 1's control at step k
J1   = get_cost(sol, 1)
```
