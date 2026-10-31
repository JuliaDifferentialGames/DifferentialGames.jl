# Working with Solutions

The `solve` function returns a `GNEPSolution` object containing the computed Nash equilibrium trajectories, costs, and strategy.

## Basic Access

```julia
sol = solve(game, FNELQ())

# Check convergence
sol.converged          # Bool
sol.iterations         # Int — number of solver iterations
sol.solve_time         # Float64 — wall time in seconds
sol.equilibrium_type   # Symbol — :FeedbackNash, :OpenLoopNash, :GeneralizedNash, :Approximate or :Unknown

# Per-player costs
get_cost(sol, 1)       # total cost for player 1
get_costs(sol)         # Vector{Float64} — all players, ordered by player id

# Per-player trajectories
traj = get_trajectory(sol, 1)
```

## Trajectory Objects

Each `Trajectory` holds a player's state and control history:

```julia
traj = get_trajectory(sol, 1)

traj.player_id          # which player
traj.states             # Matrix (n × N+1), one column per time node.
                        #   LQGameProblem: the shared state.
                        #   PDGNEProblem: the JOINT state (all players stacked), identical for every player.
traj.controls           # Matrix (mᵢ × N): this player's own controls
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

For a pure feedback law from `FNELQ` the closed loop is `u_i(k) = -get_gain(strat, i, k) * x(k)`; this is verified numerically in the [LQ tutorial](../tutorials/lq_games.md). (There is no `:feedback_gains` entry in `solver_info`; use `get_gain`.)

### OpenLoopStrategy

An `OpenLoopStrategy` stores the nominal control sequence:

```julia
strat = get_strategy(sol)
u_nom = get_nominal_control(strat, i, k)   # control for player i at step k
```

## Solver-Specific Information

Additional output is stored in `sol.solver_info::Dict{Symbol,Any}`. The keys below were read off real solutions; they may change between versions:

```julia
# FNELQ
sol.solver_info[:min_rcond]              # worst reciprocal condition number of the gain systems
sol.solver_info[:rcond_history]
sol.solver_info[:max_P_norm]
sol.solver_info[:cost_to_go_matrices]
sol.solver_info[:backward_pass_time]; sol.solver_info[:forward_pass_time]

# iLQGames
sol.solver_info[:cost_history]               # total cost per iteration
sol.solver_info[:trajectory_change_history]  # max trajectory change per iteration
sol.solver_info[:η_history]                  # accepted line-search step per iteration
sol.solver_info[:final_cost_per_player]
sol.solver_info[:open_loop_strategy]         # converged nominal (X*, U*)

# ALGAMES
sol.solver_info[:con_vio]         # constraint violation per outer iteration
sol.solver_info[:opt_vio]         # stationarity residual per outer iteration
sol.solver_info[:dyn_vio]         # dynamics residual per outer iteration
sol.solver_info[:dual_variables]  # final multipliers
sol.solver_info[:history]
```

## Rolling Out a Strategy

`rollout_strategy(dynamics, x0, strategy, p)` integrates the dynamics under a strategy (see its docstring for the keyword arguments). It is defined on the dynamics object (`game.dynamics`), not on the game.

## Checking the Solution Quality

```julia
# Cost comparison across solvers
sol1 = solve(game, iLQGames())
sol2 = solve(game, ALGAMES())

for i in 1:num_players(game)
    println("Player $i: iLQGames=$(round(get_cost(sol1,i);digits=4))  ALGAMES=$(round(get_cost(sol2,i);digits=4))")
end
```

Costs are comparable across solvers only for the same problem. `converged` is the solver's own stopping test, not a certificate of equilibrium; for constrained games also evaluate the constraints directly (see the [Constrained Games tutorial](../tutorials/constrained_games.md)). Two different local solvers (or two option sets) can return different solutions of the same problem.

## Reported costs

Costs reported by the solvers include a factor ``\tfrac12``: ``J_i = \tfrac12[\sum_k \ell_i(x_k,u_k) + \phi_i(x_N)]`` for LQ costs (confirmed by comparison with the LQR Riccati solution in the LQ tutorial).

## First-Step State (for inverse game solvers)

For inverse game inference, `first_step_state` extracts the joint state after one dynamics step:

```julia
x1 = first_step_state(sol)   # joint state at k=2
```
