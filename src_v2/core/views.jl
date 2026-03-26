module Views

using Random
using ..Kernel
using ..Capabilities: has_action_mask, has_public_observation
using ..Exact
using ..Runtime

export PlayerView
export AbstractEnvView, DecisionEnvView, SimultaneousEnvView, ChanceEnvView, TerminalEnvView
export GameEnv, FastGameEnv
export reset!, step!, current_view, player_view, observe_env
export reset_state!, step_state!, current_node

struct PlayerView{O,A,M}
    player::Int
    observation::O
    legal_actions::A      # nothing => not applicable / not acting
    legal_action_mask::M  # nothing => unavailable / not applicable
end

abstract type AbstractEnvView end

struct DecisionEnvView{PO,PV,I} <: AbstractEnvView
    public_observation::PO
    player_view::PV
    info::I
    terminated::Bool
    truncated::Bool
end

struct SimultaneousEnvView{PO,PV,I,AP} <: AbstractEnvView
    active_players::AP
    player_views::PV
    info::I
    public_observation::PO
    terminated::Bool
    truncated::Bool
end

struct ChanceEnvView{PO,I} <: AbstractEnvView
    public_observation::PO
    info::I
    terminated::Bool
    truncated::Bool
end

struct TerminalEnvView{PO,I} <: AbstractEnvView
    public_observation::PO
    info::I
    terminated::Bool
    truncated::Bool
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

"""
Low-allocation environment wrapper for training loops.

Use:
- `current_node(env)`
- `env.state`
- `reset_state!(env)`
- `step_state!(env, action)`

Avoids `PlayerView`/`EnvView` construction.
"""
mutable struct FastGameEnv{G<:Kernel.AbstractFixedGame,S,R,L}
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
                 limit = Runtime.default_episode_limit(game))
    s = Kernel.init_state(game, rng)
    t = Kernel.is_terminal(game, s)
    return GameEnv{typeof(game),typeof(s),typeof(rng),typeof(limit)}(
        game, s, rng, limit, 0, t, false
    )
end

function FastGameEnv(game::Kernel.AbstractFixedGame;
                     rng::AbstractRNG = Random.default_rng(),
                     limit = Runtime.default_episode_limit(game))
    s = Kernel.init_state(game, rng)
    t = Kernel.is_terminal(game, s)
    return FastGameEnv{typeof(game),typeof(s),typeof(rng),typeof(limit)}(
        game, s, rng, limit, 0, t, false
    )
end

@inline function _maybe_public_obs(game, state)
    has_public_observation(typeof(game)) || return nothing
    return Exact.public_observation(game, state)
end

@inline function _active_players(game, state, nk::Kernel.NodeKind)
    if nk === Kernel.DECISION
        return (Kernel.current_player(game, state),)
    elseif nk === Kernel.SIMULTANEOUS
        return Kernel.active_players(game, state)
    else
        return ()
    end
end

@inline _is_acting_player(active_players, p::Int) = p in active_players

@inline function _player_view(game, state, p::Int, acting::Bool)
    obs = Kernel.observe(game, state, p)

    if !acting
        return PlayerView(p, obs, nothing, nothing)
    end

    legal = Kernel.legal_actions(game, state, p)
    mask = nothing

    if Kernel.action_mode(typeof(game)) === Kernel.IndexedActions
        Kernel.validate_indexed_action_contract(game, state, p)
        if has_action_mask(typeof(game))
            mask = Kernel.legal_action_mask(game, state, p)
        end
    elseif has_action_mask(typeof(game))
        mask = Kernel.legal_action_mask(game, state, p)
    end

    return PlayerView(p, obs, legal, mask)
end

function player_view(env::GameEnv, p::Int)
    game = env.game
    state = env.state
    nk = Kernel.node_kind(game, state)
    aps = _active_players(game, state, nk)
    return _player_view(game, state, p, _is_acting_player(aps, p))
end

function current_view(env::GameEnv)
    game = env.game
    state = env.state
    nk = Kernel.node_kind(game, state)

    if nk === Kernel.DECISION
        p = Kernel.current_player(game, state)
        return _player_view(game, state, p, true)
    elseif nk === Kernel.SIMULTANEOUS
        aps = Kernel.active_players(game, state)
        return Tuple(_player_view(game, state, p, true) for p in aps)
    else
        return nothing
    end
end

"""
Convenience adapter only. Not intended for hot-path training loops.

Returns a node-specific view type instead of a broad union-heavy struct.
"""
function observe_env(env::GameEnv; include_all_players::Bool = false, info = Kernel.default_info())
    game = env.game
    state = env.state
    nk = Kernel.node_kind(game, state)

    terminated = Kernel.is_terminal(game, state)
    truncated = !terminated && Runtime.is_truncated(env.limit, env.step_count)
    pub = _maybe_public_obs(game, state)

    if nk === Kernel.DECISION
        p = Kernel.current_player(game, state)
        pv = _player_view(game, state, p, true)
        return DecisionEnvView(pub, pv, info, terminated, truncated)

    elseif nk === Kernel.SIMULTANEOUS
        aps = Kernel.active_players(game, state)
        pviews = if include_all_players
            Tuple(_player_view(game, state, p, _is_acting_player(aps, p)) for p in Kernel.player_ids(game))
        else
            Tuple(_player_view(game, state, p, true) for p in aps)
        end
        return SimultaneousEnvView(aps, pviews, info, pub, terminated, truncated)

    elseif nk === Kernel.CHANCE
        return ChanceEnvView(pub, info, terminated, truncated)

    else
        return TerminalEnvView(pub, info, true, false)
    end
end

function reset!(env::GameEnv)
    env.state = Kernel.init_state(env.game, env.rng)
    env.step_count = 0
    env.terminated = Kernel.is_terminal(env.game, env.state)
    env.truncated = !env.terminated && Runtime.is_truncated(env.limit, env.step_count)
    return observe_env(env)
end

function step!(env::GameEnv, action)
    env.terminated &&
        throw(ArgumentError("Cannot step a terminal state; call reset! first."))
    env.truncated &&
        throw(ArgumentError("Cannot step a truncated episode; call reset! first."))

    next_state, rewards, terminated, info =
        Kernel.step_with_info(env.game, env.state, action, env.rng)

    env.state = next_state
    env.step_count += 1
    env.terminated = terminated
    env.truncated = !terminated && Runtime.is_truncated(env.limit, env.step_count)

    return observe_env(env; info = info), rewards, env.terminated, env.truncated
end

# ------------------------------------------------------------------
# Fast path API
# ------------------------------------------------------------------

@inline current_node(env::FastGameEnv) = Kernel.node_kind(env.game, env.state)

function reset_state!(env::FastGameEnv)
    env.state = Kernel.init_state(env.game, env.rng)
    env.step_count = 0
    env.terminated = Kernel.is_terminal(env.game, env.state)
    env.truncated = !env.terminated && Runtime.is_truncated(env.limit, env.step_count)
    return env.state
end

function step_state!(env::FastGameEnv, action)
    env.terminated &&
        throw(ArgumentError("Cannot step a terminal state; call reset_state! first."))
    env.truncated &&
        throw(ArgumentError("Cannot step a truncated episode; call reset_state! first."))

    next_state, rewards, terminated, info =
        Kernel.step_with_info(env.game, env.state, action, env.rng)

    env.state = next_state
    env.step_count += 1
    env.terminated = terminated
    env.truncated = !terminated && Runtime.is_truncated(env.limit, env.step_count)

    return next_state, rewards, env.terminated, env.truncated, info
end

end