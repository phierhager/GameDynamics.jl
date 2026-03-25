module Views

using Random
using ..Kernel
using ..Capabilities: has_action_mask, has_public_observation
using ..Exact
using ..Runtime

export PlayerView, EnvView, GameEnv
export reset!, step!, current_view, player_view, observe_env

struct PlayerView{O,A,M}
    player::Int
    observation::O
    legal_actions::A
    legal_action_mask::M
end

struct EnvView{AP,PO,PV,I}
    node::Kernel.NodeKind
    current_player::Union{Nothing, Int}
    active_players::AP
    public_observation::PO
    player_views::PV
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
end

function GameEnv(game::Kernel.AbstractFixedGame;
                 rng::AbstractRNG = Random.default_rng(),
                 limit = Runtime.default_episode_limit(game))
    s = Kernel.init_state(game, rng)
    return GameEnv{typeof(game), typeof(s), typeof(rng), typeof(limit)}(game, s, rng, limit, 0)
end

function _maybe_public_obs(game, state)
    if has_public_observation(typeof(game)) === Val(true)
        return Exact.public_observation(game, state)
    end
    return nothing
end

function player_view(env::GameEnv, p::Int)
    game = env.game
    state = env.state
    mask = has_action_mask(typeof(game)) === Val(true) ?
        Kernel.legal_action_mask(game, state, p) : nothing
    return PlayerView(
        p,
        Kernel.observe(game, state, p),
        Kernel.legal_actions(game, state, p),
        mask,
    )
end

function current_view(env::GameEnv)
    game = env.game
    state = env.state
    nk = Kernel.node_kind(game, state)

    if nk == Kernel.DECISION
        p = Kernel.current_player(game, state)
        return player_view(env, p)
    elseif nk == Kernel.SIMULTANEOUS
        return Tuple(player_view(env, p) for p in Kernel.active_players(game, state))
    else
        return nothing
    end
end

"""
Convenience adapter only. Not intended for hot-path training loops.
"""
function observe_env(env::GameEnv; include_all_players::Bool = false, info = nothing, truncated::Bool = false)
    game = env.game
    state = env.state
    nk = Kernel.node_kind(game, state)

    cp = nk == Kernel.DECISION ? Kernel.current_player(game, state) : nothing
    aps = nk == Kernel.SIMULTANEOUS ? Kernel.active_players(game, state) : ()

    pviews = if include_all_players
        Tuple(player_view(env, p) for p in Kernel.player_ids(game))
    elseif nk == Kernel.DECISION
        (player_view(env, cp),)
    elseif nk == Kernel.SIMULTANEOUS
        Tuple(player_view(env, p) for p in aps)
    else
        ()
    end

    return EnvView(
        nk,
        cp,
        aps,
        _maybe_public_obs(game, state),
        pviews,
        info,
        Kernel.is_terminal(game, state),
        truncated,
    )
end

function reset!(env::GameEnv)
    env.state = Kernel.init_state(env.game, env.rng)
    env.step_count = 0
    return observe_env(env)
end

function step!(env::GameEnv, action)
    next_state, rewards, terminated = Kernel.step(env.game, env.state, action, env.rng)
    env.state = next_state
    env.step_count += 1
    truncated = !terminated && Runtime.is_truncated(env.limit, env.step_count)
    return observe_env(env; truncated = truncated), rewards, terminated, truncated
end

end