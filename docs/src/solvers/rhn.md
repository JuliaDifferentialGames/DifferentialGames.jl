# RecedingHorizonNash

**Receding Horizon Nash** solver wrapper. Produces a closed-loop Nash strategy via the receding-horizon (model predictive control) framework.

## Overview

The RHN solver wraps any `GameSolver` to produce a closed-loop Nash strategy. At each simulation step:

1. A sub-game is created with the current state as initial condition
2. The sub-game is solved using the inner solver
3. Only the first optimal control is applied
4. The solution is shifted and used as a warm-start for the next sub-problem

This implements the standard MPC (Model Predictive Control) approach for differential games, providing feedback control while using potentially open-loop solvers internally.

## Algorithm

For t = 1, …, N_sim:

```
1. sub_game   = with_initial_state(game_window, X[:,t])    # O(1) patch
2. sol        = solve(sub_game, inner_solver; warmstart=ws)   # any solver
3. u_t        = apply_strategy(sol.strategy, X[:,t], 1)    # first action
4. X[:,t+1]   = rollout_step(dyn, X[:,t], u_t, ...)        # propagate
5. ws         = shift_warmstart(sol)                      # shift by 1 step
```

The shift-and-warm-start step follows the TinyMPC / Mattingley et al. convention:
- Drop the first element of the solution
- Repeat the last element
- Use this as the initial iterate for the next sub-problem

## Usage

```julia
# Minimal - wraps any solver
solver = RecedingHorizonNash(FNELQ())
solver = RecedingHorizonNash(iLQGames())
solver = RecedingHorizonNash(ALGAMES())

# With options:
solver = RecedingHorizonNash(iLQGames();
    warm_start   = true,
    verbose_inner = false
)

# Solve a receding horizon problem
prob = RecedingHorizonNashProblem(horizon_game, x0, 50)
sol  = solve(prob, solver)
```

## Solver Type Documentation

```@docs
RecedingHorizonNash
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `inner_solver` | Required | Any `GameSolver` (FNELQ, iLQGames, ALGAMES, etc.) |
| `warm_start` | `true` | Whether to shift the previous solution as warm-start for the next solve |
| `verbose_inner` | `false` | Whether to pass `verbose=true` to inner solves |

## Capabilities

```@docs
solver_capabilities(::Type{RecedingHorizonNash})
```

## Problem Type

RHN operates on `RecedingHorizonNashProblem{T}`, which wraps a standard `GameProblem` and specifies:
- The full simulation horizon
- The prediction window length
- The initial state

## Warm-Start Shifting

After each sub-solve, the strategy is shifted by one step:

### For FeedbackStrategy:
- Gains, feedforward, and nominal trajectory are all shifted left by 1 step
- The last element is repeated
- This provides a hot-start that is valid (feasible dynamics) and near the new optimal point

### For OpenLoopStrategy:
- Control sequences are shifted left by 1 step
- The last control is repeated

### Effect on Different Solvers:
- **FNELQ**: Warm-start is structurally accepted but has no effect since FNELQ solves exactly in one backward pass
- **iLQGames/ALGAMES**: Warm-start significantly reduces inner solve iterations by starting near the previous solution

## Benefits

1. **Closed-loop control**: Produces a feedback policy even when using open-loop solvers internally
2. **Robustness**: Re-optimizes at each step, providing robustness to disturbances and model errors
3. **Efficiency**: Warm-starting from the previous solution significantly reduces computation time for nonlinear solvers
4. **Flexibility**: Can wrap any solver, allowing the same inner solver to be used for both finite-horizon and receding-horizon problems

## Applications

- **Real-time control**: Where a closed-loop strategy is required but only open-loop solvers are available
- **Adaptive control**: Where the system model or costs may change slowly over time
- **Long-horizon problems**: Where solving the full problem is infeasible, but a receding-horizon approach is tractable
- **Tracking problems**: Where the goal is to track a moving target or reference trajectory

## Practical Considerations

### Prediction Horizon Selection

The prediction horizon should be:
- Long enough to capture the dynamics and constraints
- Short enough to be computationally tractable
- Typically 5-20 steps for most applications

### Warm-Start Benefits

Warm-starting can reduce:
- iLQGames iterations by 50-80%
- ALGAMES iterations by 30-60%
- Total solve time by 40-70% for nonlinear problems

### Initial State Handling

The first solve (t=1) has no warm-start. Subsequent solves use the shifted solution from the previous step.

## References

Mattingley, J., Wang, A., Boyd, S. (2011). *Receding Horizon Control.*

Mahajan et al. (2026). *Conic-TinyMPC* (shift warm-start convention).

Laine et al. (2023). *GFNE (generalized feedback Nash equilibrium)*.

See also the implementation notes in [DifferentialGamesBaseSolvers.jl](https://github.com/JuliaDifferentialGames/DifferentialGamesBaseSolvers.jl).
