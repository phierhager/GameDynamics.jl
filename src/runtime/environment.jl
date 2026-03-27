module RuntimeEnvironment

using Random
using ..Kernel
using ..Spec

export AbstractEpisodeLimit
export UnlimitedHorizon
export StepLimit

export default_episode_limit
export is_truncated
export remaining_steps

export default_step_info
export step_with_info

export PlayerSnapshot
export GameEnv

export current_state
export current_node
export current_observation
export current_observations
export observe_player

export terminated
export truncated
export done

export reset!
export step!

# ----------------------------------------------------------------------
# Episode limit semantics
# ----------------------------------------------------------------------

abstract type AbstractEpisodeLimit end

"""
No truncation from a runtime step budget.
"""
struct UnlimitedHorizon <: AbstractEpisodeLimit end

"""
Truncate after `max_steps` environment transitions.
"""
struct StepLimit <: AbstractEpisodeLimit
    max_steps::Int
    function StepLimit(max_steps::Int)
        max_steps > 0 || throw(ArgumentError("max_steps must be positive, got $max_steps."))
        new(max_steps)
    end
end

function default_episode_limit(game::Kernel.AbstractGame)
    max_steps = Spec.game_spec(game).max_steps
    return isnothing(max_steps) ? UnlimitedHorizon() : StepLimit(max_steps)
end

is_truncated(::UnlimitedHorizon, step_count::Int) = false
is_truncated(limit::StepLimit, step_count::Int) = step_count >= limit.max_steps

remaining_steps(::UnlimitedHorizon, step_count::Int) = typemax(Int)
remaining_steps(limit::StepLimit, step_count::Int) = max(limit.max_steps - step_count, 0)

# ----------------------------------------------------------------------
# Step-info extension hook
# ----------------------------------------------------------------------

"""
Default runtime metadata returned by `step!`.

Games or wrappers that want richer runtime diagnostics can specialize
`default_step_info(game)` and/or `step_with_info(...)`.
"""
default_step_info(::Kernel.AbstractGame) = NamedTuple()

"""
Stable runtime stepping hook that augments `Kernel.step` with:
- `terminated`
- `info`

Returns:
`next_state, rewards, terminated, info`
"""
function step_with_info(game::Kernel.AbstractGame,
                        state,
                        action,
                        rng::AbstractRNG = Random.default_rng())
    next_state, rewards = Kernel.step(game, state, action, rng)
    term = Kernel.is_terminal(game, next_state)
    return next_state, rewards, term, default_step_info(game)
end

# ----------------------------------------------------------------------
# Player-facing snapshot
# ----------------------------------------------------------------------

"""
Runtime view for a single player at the current environment state.

Fields:
- `player`: player id
- `observation`: player's current observation
- `legal_actions`: current legal actions, or `nothing` if the player is inactive
- `legal_action_mask`: optional action mask, or `nothing`
"""
struct PlayerSnapshot{O,A,M}
    player::Int
    observation::O
    legal_actions::A
    legal_action_mask::M
end

# ----------------------------------------------------------------------
# Mutable environment wrapper
# ----------------------------------------------------------------------

mutable struct GameEnv{G<:Kernel.AbstractGame,S,RNG<:AbstractRNG,L<:AbstractEpisodeLimit}
    game::G
    state::S
    rng::RNG
    limit::L
    step_count::Int
    terminated::Bool
    truncated::Bool
end

function GameEnv(game::Kernel.AbstractGame;
                 rng::AbstractRNG = Random.default_rng(),
                 limit::AbstractEpisodeLimit = default_episode_limit(game))
    s = Kernel.init_state(game, rng)
    t = Kernel.is_terminal(game, s)
    tr = !t && is_truncated(limit, 0)

    return GameEnv{typeof(game),typeof(s),typeof(rng),typeof(limit)}(
        game, s, rng, limit, 0, t, tr
    )
end

# ----------------------------------------------------------------------
# Basic accessors
# ----------------------------------------------------------------------

current_state(env::GameEnv) = env.state
current_node(env::GameEnv) = Kernel.node_kind(env.game, env.state)

terminated(env::GameEnv) = env.terminated
truncated(env::GameEnv) = env.truncated
done(env::GameEnv) = env.terminated || env.truncated

current_observation(env::GameEnv, player::Int) =
    Kernel.observe(env.game, env.state, player)

current_observations(env::GameEnv) =
    ntuple(i -> Kernel.observe(env.game, env.state, i), Kernel.num_players(env.game))

# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

@inline function _maybe_action_mask(game, state, p::Int)
    Kernel.has_action_mask(typeof(game)) || return nothing
    return Kernel.legal_action_mask(game, state, p)
end

@inline function _update_flags!(env::GameEnv)
    env.terminated = Kernel.is_terminal(env.game, env.state)
    env.truncated = !env.terminated && is_truncated(env.limit, env.step_count)
    return env
end

@inline function _require_valid_player(game, p::Int)
    p in Kernel.player_ids(game) || throw(ArgumentError("Invalid player id $p."))
    return p
end

@inline function _reset_env!(env::GameEnv)
    env.state = Kernel.init_state(env.game, env.rng)
    env.step_count = 0
    _update_flags!(env)
    return env
end

@inline function _step_env!(env::GameEnv, action)
    env.terminated &&
        throw(ArgumentError("Cannot step a terminal state; call `reset!` first."))
    env.truncated &&
        throw(ArgumentError("Cannot step a truncated episode; call `reset!` first."))

    next_state, rewards, terminated_, info =
        step_with_info(env.game, env.state, action, env.rng)

    env.state = next_state
    env.step_count += 1
    env.terminated = terminated_
    env.truncated = !terminated_ && is_truncated(env.limit, env.step_count)

    return rewards, info
end

# ----------------------------------------------------------------------
# Player snapshot API
# ----------------------------------------------------------------------

"""
Return the current runtime snapshot for player `p`.

If the player is inactive at the current node:
- observation is still provided
- `legal_actions` is `nothing`
- `legal_action_mask` is `nothing`
"""
function observe_player(env::GameEnv, p::Int)
    game = env.game
    state = env.state

    _require_valid_player(game, p)

    acting = p in Kernel.acting_players(game, state)
    obs = Kernel.observe(game, state, p)

    if !acting
        return PlayerSnapshot(p, obs, nothing, nothing)
    end

    legal = Kernel.legal_actions(game, state, p)
    mask = _maybe_action_mask(game, state, p)
    return PlayerSnapshot(p, obs, legal, mask)
end

# ----------------------------------------------------------------------
# Public environment API
# ----------------------------------------------------------------------

"""
Reset the environment to a fresh initial state.

Returns the new current state.
"""
function reset!(env::GameEnv)
    _reset_env!(env)
    return env.state
end

"""
Step the environment with a kernel-level action.

Returns:
`(next_state, rewards, terminated, truncated, info)`
"""
function step!(env::GameEnv, action)
    rewards, info = _step_env!(env, action)
    return env.state, rewards, env.terminated, env.truncated, info
end

end