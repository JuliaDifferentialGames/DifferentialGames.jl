# Working with Solutions

The `solve` function returns a `GNEPSolution` object containing the computed Nash equilibrium trajectories, costs, and strategy.

## Basic Access

```julia
sol = solve(game, FNELQ())

# Check convergence
sol.converged          # Bool
sol.iterations         # Int — number of solver iterations
sol.solve_time         # Float64 — wall time in seconds
sol.equilibrium_type   # Symbol — :FeedbackNash or :OpenLoopNash

# Per-player costs
get_cost(sol, 1)       # total cost for player 1
get_costs(sol)         # Dict{Int, Float64} — all players

# Per-player trajectories
traj = get_trajectory(sol, 1)
```

## Trajectory Objects

Each `Trajectory` holds a player's state and control history:

```julia
traj = get_trajectory(sol, 1)

traj.player_id          # which player
traj.states             # Matrix (n × N+1): states[col, k] = state at step k
traj.controls           # Matrix (m × N):   controls[col, k] = control at step k
traj.cost               # Float64: this player's total cost
traj.times              # Vector (N+1): time at each step (if available)
```

Indexing:

```julia
x_initial = traj.states[:, 1]       # initial state
x_final   = traj.states[:, end]     # terminal state
u_at_k    = traj.controls[:, k]     # control at step k
```

## Strategies

When a feedback strategy is stored (FNELQ, iLQGames), access it via:

```julia
has_strategy(sol)          # Bool — strategy is stored
is_feedback(sol)           # Bool — it's a feedback strategy
is_open_loop_solution(sol) # Bool — it's an open-loop strategy

strat = get_strategy(sol)  # FeedbackStrategy or OpenLoopStrategy
```

### FeedbackStrategy

A `FeedbackStrategy` stores time-varying gain matrices for all players:

```julia
strat = get_strategy(sol)

# For player i at time step k:
# u_i(k) = -K_i(k) * (x(k) - x̄(k)) + ū_i(k)
K, x̄, ū = get_gain(strat, i, k), get_nominal_state(strat, k), get_feedforward(strat, i, k)
```

FNELQ also stores gains directly in solver_info:

```julia
gains = sol.solver_info[:feedback_gains]
# gains[i][k] is the (m_i × n) gain matrix for player i at step k
```

### OpenLoopStrategy

An `OpenLoopStrategy` stores the nominal control sequence:

```julia
strat = get_strategy(sol)
u_nom = get_nominal_control(strat, i, k)   # control for player i at step k
```

## Solver-Specific Information

Additional solver output is stored in `sol.solver_info`:

```julia
# FNELQ
sol.solver_info[:feedback_gains]   # Vector{Vector{Matrix}} — gain matrices
sol.solver_info[:costs]            # per-player costs (same as get_costs(sol))

# iLQGames
sol.solver_info[:cost_history]     # Vector{Dict} — cost per player per iteration
sol.solver_info[:Δ_history]        # Vector{Float64} — trajectory change per iteration

# ALGAMES
sol.solver_info[:constraint_violation]  # final primal feasibility
sol.solver_info[:dual_residual]         # final dual feasibility
sol.solver_info[:multipliers]           # final Lagrange multipliers
```

## Rolling Out a Strategy

To simulate the closed-loop system under a computed strategy:

```julia
x0_new = [2.0, 0.0, -2.0, 0.0]   # different initial condition
strat  = get_strategy(sol)

# rollout_strategy integrates the dynamics under the feedback strategy
traj_new = rollout_strategy(game, strat, x0_new)
```

## Checking the Solution Quality

```julia
# Cost comparison across solvers
sol1 = solve(game, iLQGames())
sol2 = solve(game, ALGAMES())

for i in 1:num_players(game)
    println("Player $i: iLQGames=$(round(get_cost(sol1,i);digits=4))  ALGAMES=$(round(get_cost(sol2,i);digits=4))")
end
```

## First-Step State (for inverse game solvers)

For inverse game inference, `first_step_state` extracts the joint state after one dynamics step:

```julia
x1 = first_step_state(sol)   # joint state at k=2
```
