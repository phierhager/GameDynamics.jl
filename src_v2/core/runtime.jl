module Runtime

using Random
using ..Kernel
using ..Spec

export AbstractEpisodeLimit, NoLimit, StepLimit
export default_episode_limit, is_truncated

export default_step_info, step_with_info

export PlayerSnapshot
export GameEnv
export current_node, reset!, step!, observe_player

abstract type AbstractEpisodeLimit end

struct NoLimit <: AbstractEpisodeLimit end

struct StepLimit <: AbstractEpisodeLimit
    max_steps::Int
    function StepLimit(max_steps::Int)
        max_steps > 0 || throw(ArgumentError("max_steps must be positive, got $max_steps."))
        new(max_steps)
    end
end

function default_episode_limit(game::Kernel.AbstractGame)
    max_steps = Spec.game_spec(game).max_steps
    return isnothing(max_steps) ? NoLimit() : StepLimit(max_steps)
end

is_truncated(::NoLimit, step_count::Int) = false
is_truncated(limit::StepLimit, step_count::Int) = step_count >= limit.max_steps

default_step_info(::Kernel.AbstractGame) = NamedTuple()

function step_with_info(game::Kernel.AbstractGame, state, action,
                        rng::AbstractRNG = Random.default_rng())
    next_state, rewards = Kernel.step(game, state, action, rng)
    terminated = Kernel.is_terminal(game, next_state)
    return next_state, rewards, terminated, default_step_info(game)
end

struct PlayerSnapshot{O,A,M}
    player::Int
    observation::O
    legal_actions::A
    legal_action_mask::M
end

mutable struct GameEnv{G<:Kernel.AbstractFixedGame,S,R,L}
    game::G
    state::S
    rng::R
    limit::L
    step_count::Int
    terminated::Bool
    truncated::Bool
end

function GameEnv(game::Kernel.AbstractFixedGame;
                 rng::AbstractRNG = Random.default_rng(),
                 limit = default_episode_limit(game))
    s = Kernel.init_state(game, rng)
    t = Kernel.is_terminal(game, s)
    tr = !t && is_truncated(limit, 0)
    return GameEnv{typeof(game),typeof(s),typeof(rng),typeof(limit)}(
        game, s, rng, limit, 0, t, tr
    )
end

@inline function _active_players(game, state, nk::Kernel.NodeKind)
    if nk == Kernel.DECISION
        return (Kernel.current_player(game, state),)
    elseif nk == Kernel.SIMULTANEOUS
        return Kernel.active_players(game, state)
    else
        return ()
    end
end

@inline _is_acting_player(active_players, p::Int) = p in active_players

@inline function _maybe_action_mask(game, state, p::Int)
    Kernel.has_action_mask(typeof(game)) || return nothing
    return Kernel.legal_action_mask(game, state, p)
end

@inline function _update_flags!(env)
    env.terminated = Kernel.is_terminal(env.game, env.state)
    env.truncated = !env.terminated && is_truncated(env.limit, env.step_count)
    return env
end

@inline function _reset_env!(env)
    env.state = Kernel.init_state(env.game, env.rng)
    env.step_count = 0
    _update_flags!(env)
    return env
end

@inline function _step_env!(env, action)
    env.terminated &&
        throw(ArgumentError("Cannot step a terminal state; call reset! first."))
    env.truncated &&
        throw(ArgumentError("Cannot step a truncated episode; call reset! first."))

    next_state, rewards, terminated, info =
        step_with_info(env.game, env.state, action, env.rng)

    env.state = next_state
    env.step_count += 1
    env.terminated = terminated
    env.truncated = !terminated && is_truncated(env.limit, env.step_count)

    return rewards, info
end

@inline current_node(env::GameEnv) = Kernel.node_kind(env.game, env.state)

function observe_player(env::GameEnv, p::Int)
    game = env.game
    state = env.state
    nk = Kernel.node_kind(game, state)
    aps = _active_players(game, state, nk)
    acting = _is_acting_player(aps, p)

    obs = Kernel.observe(game, state, p)
    if !acting
        return PlayerSnapshot(p, obs, nothing, nothing)
    end

    legal = Kernel.legal_actions(game, state, p)
    mask = _maybe_action_mask(game, state, p)
    return PlayerSnapshot(p, obs, legal, mask)
end

function reset!(env::GameEnv)
    _reset_env!(env)
    return env.state
end

function step!(env::GameEnv, action)
    rewards, info = _step_env!(env, action)
    return env.state, rewards, env.terminated, env.truncated, info
end

end