# MADDPG

`MADDPGSolver` configures multi-agent deep deterministic policy gradient training ([Lowe2017](@cite)): each agent has a decentralized **actor** (its own observation → its action) and a **centralized critic** (all observations and actions → value), trained from a shared replay buffer with soft target updates.

## Usage

```julia
using DifferentialGamesMARL, Random

rng = MersenneTwister(1)
nav = cooperative_navigation(; n_agents = 3)
env = MADDPGEnv(nav.game; obs_fn = nav.obs_fn,
                reset_fn = () -> nav.sample_start(rng = rng))

solver = MADDPGSolver(; episodes = 30, warmup = 50, batch_size = 32,
                        buffer_size = 2000, eval_every = 10, hidden = 16)

agents, history, agent_history, evals = train!(solver, env; rng = rng, verbose = false)
```

(This 30-episode pipeline check took about 13 s. Use the default `episodes = 25_000` for a real run; it takes hours.)

## Return values of `train!`

| Value | Content |
|-------|---------|
| `agents` | `Vector{Agent}`; with `track_best = true` the best policy by greedy evaluation, otherwise the last |
| `history` | per-episode **training** rewards summed over agents and steps (includes exploration noise) |
| `agent_history` | `(episodes, n_agents)` matrix of per-agent training rewards |
| `evals` | named tuple: `episodes`, `rewards` (greedy evaluations, per agent), `q_magnitude`, `noise`, `wall`, `best_episode`, `best_score`, `final_agents`, `eval_starts`, `guide_log`, `wall_total` |

Learning curves should show `evals.rewards`: greedy evaluation episodes without exploration noise, averaged over `n_eval_episodes` held-out start states drawn once before training and replayed at every evaluation point.

## Hyperparameters

`MADDPGSolver` is a `@kwdef` struct; the defaults are starting points for cooperative navigation, not tuned optima.

| Field | Default | Meaning |
|-------|---------|---------|
| `episodes` | `25_000` | training episodes |
| `batch_size`, `buffer_size` | `256`, `100_000` | minibatch and replay capacity |
| `γ`, `τ` | `0.90`, `0.01` | discount and soft target-update rate |
| `lr_actor`, `lr_critic`, `hidden` | `5e-4`, `1e-3`, `64` | learning rates and hidden width |
| `noise_std`, `noise_decay`, `noise_floor` | `0.2`, `0.9999`, `0.02` | exploration noise schedule; the floor keeps exploration alive for the whole run |
| `update_every`, `warmup` | `2`, `1000` | steps between updates; random-action warm-up steps |
| `eval_every`, `n_eval_episodes` | `50`, `1` | greedy evaluation cadence and number of held-out starts |
| `grad_clip` | `1.0` | bound on any single gradient update (0 disables) |
| `checkpoint_every`, `checkpoint_dir` | `0`, `""` | serialize agents every N episodes (`.jls`) |
| `track_best` | `true` | return the best-evaluated policy |
| `ρ`, `guidance_builder`, `guide_every` | `0.0`, `nothing`, `50` | [guided policy search](gps.md); `ρ = 0` is plain MADDPG at no cost |

## Interpreting results

- **A trained policy is not an equilibrium.** Nothing certifies a Nash equilibrium; check exploitability separately (for example by training a best response against frozen agents: `update_agents!(…; frozen = (…))` supports a frozen opponent).
- **`track_best` and adversarial games.** A higher *summed* score is not "better" in a zero-sum or competitive game: one side scoring well means the other is losing. Set `track_best = false` for adversarial tasks such as `predator_prey`.
- **Exploration.** With the default schedule noise falls to ≈0.12 at episode 5 000 and to the 0.02 floor at 25 000. A faster decay makes actors effectively deterministic early and can stall learning.
- **Reward scale.** Rewards include barrier terms from `constraint_mode = :barrier`; use `barrier_cap` if early training is unstable.

## Lower-level API

`Agent(obs_dim, act_dim, n_agents; lr_actor, lr_critic, hidden, grad_clip)`, `ReplayBuffer`/`Transition`/`store!`, `build_actor`/`build_critic` (Flux networks), `select_actions(agents, obs; noise_std, rng)` (greedy actions plus noise, clamped to `[−1, 1]`) and `update_agents!(agents, batch, n_agents, act_dim; γ, τ, ρ, guidance, frozen)` expose the pieces of the training loop.

## References

[Lowe2017](@cite). The package's source also describes the exploration and tuning choices behind the defaults (`src/solver.jl`).
