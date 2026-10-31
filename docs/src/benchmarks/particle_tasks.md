# Multi-Agent Particle Tasks

The experiments of the MADDPG paper ([Lowe2017](@cite)) use a family of six two-dimensional particle tasks, with cooperative and competitive variants. `DifferentialGamesMARL.jl` builds four of them as ordinary Base games (so that a model-based solver, a learned policy and the [benchmark evaluator](overview.md) can all be pointed at the same problem) and does not represent the other two.

| Task in the paper | Players | Builder | Status |
|-------------------|---------|---------|--------|
| Cooperative navigation | `N` cooperating agents, `N` landmarks | [`cooperative_navigation`](#cooperative_navigation) | implemented |
| Predator–prey | `N` slower predators, 1 faster prey, optional obstacles | [`predator_prey`](#predator_prey) | implemented |
| Keep-away | `N` agents, `M` adversaries, `L` landmarks | [`keep_away`](#keep_away) | implemented (blocking instead of pushing) |
| Physical deception | `N` agents, 1 adversary, `N` landmarks | [`physical_deception`](#physical_deception) | implemented |
| Cooperative communication | speaker, listener | — | **not implemented** (needs a message channel) |
| Covert communication | Alice, Bob, Eve | — | **not implemented** (needs a message channel) |

All four implemented tasks share the same agent model, the damped planar point mass ``\dot p = v,\ \dot v = a\,u - c\,v`` with state ``[p_x, p_y, v_x, v_y]`` and control ``u\in[-1,1]^2``. Each is a `PDGNEProblem`-style game with `SeparableDynamics`, so each runs under `ALGAMES` and `MADDPGEnv`. Defaults follow the paper's horizon (25 steps of 0.1 s).

!!! note "Status"
    Everything on this page was run against the current sources. The code samples are **not executed** by the documentation build because they pull in `DifferentialGamesMARL.jl` (which depends on Flux). No MARL training result is claimed here: the training loop has no convergence test, see the [audit](https://github.com/JuliaDifferentialGames/DifferentialGames.jl/blob/main/docs/AUDIT.md).

## `cooperative_navigation`

`N` agents cover `N` landmarks. All agents share the cost ``\sum_{\ell}\min_a\|p_a-\ell\| + 0.01\|u\|^2`` and are kept apart by pairwise collision constraints. See [Environments and tasks](../marl/environments.md#cooperative_navigation) for the options.

## `predator_prey`

`N` slower predators chase one faster prey in a bounded arena; costs are adversarial, with optional obstacles and sparse (contact-only) rewards. See [Environments and tasks](../marl/environments.md#predator_prey).

## `keep_away`

> *Keep-away*: ``L`` landmarks including a target; ``N`` cooperating agents who know the target are rewarded for reaching it; ``M`` adversaries must keep them away but do not know which landmark is the target ([Lowe2017](@cite)).

```julia
using DifferentialGamesMARL

task = keep_away(; n_agents = 2, n_adversaries = 1, n_landmarks = 2, target = 1)
keys(task)   # (:game, :landmarks, :target, :goal, :n_agents, :n_adversaries,
             #  :agent_radius, :obs_fn, :sample_start)
```

| Part | Definition |
|------|------------|
| Players | `1:n_agents` cooperating agents, then the adversaries |
| Agent cost | ``\|p_i-g\| - \min_k\|q_k-g\| + 0.01\|u_i\|^2`` (own distance to the target minus the closest adversary's), `g` the target, `q` adversary positions |
| Adversary cost | ``\|q_k-g\| + 0.01\|u_k\|^2`` |
| Constraints | pairwise collision avoidance between every agent and every adversary (`d_min = 2·agent_radius`); position bounds `±arena` |
| Observation | own state, **target position (zeros for an adversary)**, every landmark, every other player's position; same length for every player (`4 + 2 + 2L + 2(n-1)`, which is 14 by default) |

**Deviation from the paper.** The particle environment lets adversaries *push* the agents away through soft contact forces. A contact force couples the agents' dynamics, which would destroy the separable structure the PD-GNEP solvers rely on, so here an adversary can only *block*: it cannot occupy the same space as an agent, but it cannot shove it. Results are therefore not directly comparable with numbers in the paper.

## `physical_deception`

> *Physical deception*: ``N`` cooperating agents and one adversary, ``N`` landmarks, one of them the target. The agents are rewarded for any of them reaching the target and for the adversary being far from it; the adversary is rewarded for reaching the target but does not know which landmark it is, so the agents must spread out to cover all of them ([Lowe2017](@cite)).

```julia
task = physical_deception(; n_agents = 2, n_landmarks = 2)             # the paper's N = 2
task = physical_deception(; n_agents = 4, n_landmarks = 4)             # the paper's N = 4
```

Agents share the cost ``\min_j\|p_j-g\| - \|q-g\|`` (the team is paid for its *best* member, which is what makes covering every landmark worthwhile), the adversary pays ``\|q-g\|``. Constraints are the agent–agent collision constraints and the arena bounds; the observation hides the target from the adversary exactly as in `keep_away`.

## Using the tasks

As learning problems:

```julia
using DifferentialGamesMARL, Random
rng  = MersenneTwister(1)
task = physical_deception()
env  = MADDPGEnv(task.game; obs_fn = task.obs_fn,
                 reset_fn = () -> task.sample_start(rng = rng))
solver = MADDPGSolver(; episodes = 25_000, track_best = false)   # competitive: do not select by summed reward
agents, history, agent_history, evals = train!(solver, env; rng = rng)
```

Use `track_best = false` here: a high *summed* reward is not better in a competitive game (see [MADDPG](../marl/maddpg.md)).

As model-based problems: the `game` field is a Base game, so a solver can be run on it directly.

```julia
using DifferentialGamesBase, DifferentialGamesBaseSolvers
sol = DifferentialGamesBase.solve(task.game, ALGAMES(; outer_iter = 20))
```

!!! warning "The two formulations are not the same game"
    A model-based Nash solver such as `ALGAMES` sees the whole game, including which landmark is the target, so the adversary effectively *knows* the target and the deception that the paper studies cannot arise. The information asymmetry lives only in `obs_fn`, which the learner uses and the model-based solver ignores. To reproduce the deception behaviour you must use the learner, or model the adversary's uncertainty explicitly (see [Uncertainty](../problem_types/uncertainty.md)).

In the configuration above (2+1 players, default parameters), `ALGAMES` returned in 24 s (`keep_away`) and 40 s (`physical_deception`) without reporting convergence; the final distances to the target were 0.26, 0.26 (agents) and 1.13 (adversary) for `keep_away`, and 0.01 and 1.06 (agents) and 0.04 (adversary) for `physical_deception`. The adversary reaches the target in the second task because it knows where it is. These are not equilibrium certificates; they only show that the games are well-posed for a model-based solver.

## Tasks that are not implemented

**Cooperative communication** (a speaker who observes the target colour and a listener who must navigate to it) and **covert communication** (Alice sends a message to Bob over a channel Eve also observes, and encodes it with a shared key) both revolve around a *message channel*: an action that does not move anything but is observed by other agents, produced in the paper through a Gumbel-softmax relaxation of discrete symbols. `MADDPGEnv` has no such channel. The physical half of cooperative communication could be built as a Base game with a custom `obs_fn`, but without the message action it would only be the listener's navigation problem, so it is not offered as a benchmark.

Covert communication is not a differential game at all (there is no physical state), and it would need a different environment type rather than a task builder.
