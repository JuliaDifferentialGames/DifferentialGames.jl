# Environments and Tasks

## `MADDPGEnv`

`MADDPGEnv(game; kwargs...)` wraps a Base `GameProblem` as a [`CommonRLInterface`](https://github.com/JuliaReinforcementLearning/CommonRLInterface.jl) environment.

| Base object | In the environment |
|-------------|--------------------|
| `game.dynamics` | state update inside `act!` (forward Euler with `dt`, or a discrete map) |
| `game.objectives` | the reward of player `i` is `−cost` |
| `game.initial_state` | restored by `reset!` (or a fresh start from `reset_fn`) |
| `game.time_horizon` | `terminated` is true after the last step |
| constraints | handled in `act!` by `constraint_mode` |

The **joint action** is `[u₁; u₂; …]` sliced by the control offsets. `act!` returns a **vector of rewards**, one per player. `observe(env)` is the joint state; `local_observations(env)` gives the per-agent observation vectors the actors consume.

### Options

| Keyword | Default | Meaning |
|---------|---------|---------|
| `constraint_mode` | `:barrier` | `:barrier` charges `barrier_weight·exp(barrier_sharpness·g)` for an inequality `g ≤ 0` (smooth and finite everywhere, so it tolerates the infeasible states early policies produce); `:penalty` charges `penalty_weight·max(0,g)`; `:none` ignores constraints |
| `barrier_weight`, `barrier_sharpness`, `barrier_cap` | `1.0`, `8.0`, `Inf` | barrier shape; use `barrier_cap` when deep violations make the exponential dominate the stage costs |
| `penalty_weight` | `10.0` | weight of the `:penalty` mode |
| `action_limit` | `1.0` | controls are clamped to `[−action_limit, action_limit]` |
| `obs_fn` | `nothing` | `state -> Vector{Vector{Float64}}`: per-agent observations (default: every agent sees the full joint state) |
| `state_projection` | `nothing` | in-place projection applied after each step, for constraints that must *never* be violated (hard walls, impassable obstacles) |
| `reset_fn` | `nothing` | returns a fresh initial state each episode, to train and evaluate over a *distribution* of starts; a Nash equilibrium is defined over the state space, a fixed start measures one path through it |
| `discrete_dynamics` | `false` | `false`: forward-Euler the continuous right-hand side; `true`: treat the dynamics as a map `x₊ = f(x,u,k)` (what `LinearDynamics` is) |
| `cost_state` | `:joint` | `:joint` passes the **joint** state to a stage cost, `:player` the player's own slice |

Using `LinearDynamics` games requires `discrete_dynamics = true`: forward-Euler-ing a map that is already a map is wrong.

## Stepping an environment

Verified against the current sources:

```julia
using DifferentialGamesMARL, Random
import CommonRLInterface as RL

rng = MersenneTwister(2)
nav = cooperative_navigation(; n_agents = 3)
env = MADDPGEnv(nav.game; obs_fn = nav.obs_fn, reset_fn = () -> nav.sample_start(rng = rng))

RL.reset!(env)
rewards = RL.act!(env, zeros(6))      # joint action for 3 agents × 2 controls  → Vector{Float64} (3)
RL.terminated(env)                    # false until the last step; the 2.5 s task has 25 steps
num_agents(env), joint_control_dim(env), length.(local_observations(env))   # (3, 6, [14, 14, 14])
```

## Built-in tasks

The four tasks below return a `NamedTuple` containing an ordinary Base `game` (a `PDGNEProblem` with separable damped double-integrator dynamics), so any model-based solver can be run on it too.

### `cooperative_navigation`

The cooperative navigation benchmark of [Lowe2017](@cite): `n_agents` point masses must cover `n_agents` landmarks without colliding. Every agent shares the same coverage cost (sum over landmarks of the distance to the *nearest* agent) plus a small control penalty; collisions are pairwise shared proximity constraints.

```julia
nav = cooperative_navigation(; n_agents = 3, dt = 0.1, tf = 2.5, accel = 3.0, damping = 0.5,
                               agent_radius = 0.15, control_penalty = 0.01, randomize = false)
keys(nav)   # (:game, :landmarks, :agent_radius, :obs_fn, :sample_start)
```

`nav.sample_start(; rng)` draws a feasible start. Per-agent observations include the landmarks, so `obs_fn` must be passed to `MADDPGEnv`.

### `predator_prey`

The competitive tag scenario: `n_predators` slower predators chase one faster prey in a soft-bounded square arena. Costs are adversarial and distance-shaped (a predator minimizes its distance to the prey; the prey maximizes the distance to the nearest predator). The arena is a per-player `state_bounds` private constraint; predator-predator collisions are shared constraints. There is deliberately no predator–prey proximity constraint.

```julia
pp = predator_prey(; n_predators = 2,
                     obstacles = [rectangle_obstacle([0.0, 0.0], 0.4, 0.4; margin = 0.15)])
keys(pp)   # (:game, :n_predators, :arena, :agent_radius, :obs_fn, :state_projection,
           #  :obstacles, :obstacles_inflated, :capture_radius, :sample_start)
env = MADDPGEnv(pp.game; obs_fn = pp.obs_fn, state_projection = pp.state_projection,
                reset_fn = () -> pp.sample_start(rng = rng))
```

The returned `state_projection` makes obstacles impassable; any positive `obstacle_barrier_scale` is pure reward shaping.

### `keep_away` and `physical_deception`

The two target-and-adversary tasks of [Lowe2017](@cite): cooperating agents that know the target landmark against adversaries that do not. The information asymmetry is in the observation (the adversaries' target slot is zeroed), and the paper's contact pushing is replaced by collision-avoidance constraints. See [Multi-agent particle tasks](../benchmarks/particle_tasks.md) for the definitions, costs and what is and is not reproduced.

```julia
ka = keep_away(; n_agents = 2, n_adversaries = 1)
pd = physical_deception(; n_agents = 4, n_landmarks = 4)
env = MADDPGEnv(pd.game; obs_fn = pd.obs_fn, reset_fn = () -> pd.sample_start(rng = rng))
```

### Polygon obstacles

`ConvexObstacle`, `polygon_obstacle(vertices; margin)`, `rectangle_obstacle(centre, width, height; margin)` and `regular_polygon_obstacle(centre, radius, n_sides; …)` build convex polygons. `clearance(obs, p)` is the signed clearance (negative inside), `is_inside`, `min_width`, `obstacle_projection`, `obstacle_constraints`, `compose_projections` and `default_obstacles` support constraint construction.

```julia
ob = rectangle_obstacle([0.0, 0.0], 0.4, 0.4; margin = 0.1)
clearance(ob, [0.0, 0.0])     # -0.3   (inside)
clearance(ob, [1.0, 0.0])     #  0.7
```

## Cost conventions

The built-in tasks write their costs against the **joint** state and the player's *own* control, and the environment's default `cost_state = :joint` matches that. Base's own expansion (used by iLQGames and by [guided policy search](gps.md)) passes a player's *own* state slice instead; see [Conventions](../ecosystem/conventions.md). A game written for one convention is silently wrong under the other, which is why `cost_state` is explicit.
