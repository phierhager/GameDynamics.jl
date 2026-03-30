module RuntimeStrategyExecution

using Random

using ..Kernel
using ..StrategyInterface
using ..StrategyProfiles
using ..ExtensiveFormInfosets

export require_profile_length
export player_context
export sample_strategy_action
export action_probability_at
export sample_joint_action

"""
Check that a strategy profile matches the player count of a game.

This lives in runtime compatibility code, not in `strategies/profiles.jl`,
because it is a profile-game compatibility check rather than an intrinsic
profile property.
"""
function require_profile_length(profile::StrategyProfiles.StrategyProfile,
                                game::Kernel.AbstractGame)
    N = Kernel.num_players(game)
    length(profile) == N || throw(ArgumentError(
        "Strategy profile length $(length(profile)) does not match number of players $N."
    ))
    return profile
end

"""
Construct the context object that `strategy` should see for `player` at `(game, state)`.

Supported built-in context kinds:
- `NoContext`
- `ObservationContext`
- `StateContext`
- `InfosetContext`

`HistoryContext` and `CustomContext` are intentionally not handled generically here,
because they require domain-specific context construction.
"""
function player_context(strategy::StrategyInterface.AbstractStrategy,
                        game::Kernel.AbstractGame,
                        state,
                        player::Int)
    kind = StrategyInterface.context_kind(strategy)

    if kind isa StrategyInterface.NoContext
        return nothing

    elseif kind isa StrategyInterface.ObservationContext
        return Kernel.observe(game, state, player)

    elseif kind isa StrategyInterface.StateContext
        return state

    elseif kind isa StrategyInterface.InfosetContext
        return ExtensiveFormInfosets.infoset(game, state, player)

    elseif kind isa StrategyInterface.HistoryContext
        throw(ArgumentError(
            "Generic runtime querying cannot construct a history context. Use a domain-specific helper."
        ))

    else
        throw(ArgumentError(
            "Generic runtime querying does not know how to construct context kind $(typeof(kind))."
        ))
    end
end

"""
Sample the action induced by `strategy` for `player` at `(game, state)`.
"""
function sample_strategy_action(strategy::StrategyInterface.AbstractStrategy,
                            game::Kernel.AbstractGame,
                            state,
                            player::Int,
                            rng::AbstractRNG = Random.default_rng())
    ctx = player_context(strategy, game, state, player)
    if ctx === nothing
        return StrategyInterface.sample_action(strategy, rng)
    else
        return StrategyInterface.sample_action(strategy, ctx, rng)
    end
end

"""
Return the action probability induced by `strategy` for `player` at `(game, state)`.
"""
function action_probability_at(strategy::StrategyInterface.AbstractStrategy,
                               game::Kernel.AbstractGame,
                               state,
                               player::Int,
                               action)
    ctx = player_context(strategy, game, state, player)
    if ctx === nothing
        return StrategyInterface.action_probability(strategy, action)
    else
        return StrategyInterface.action_probability(strategy, ctx, action)
    end
end

@inline function _check_legal_action(game, state, p::Int, a)
    legal = Kernel.legal_actions(game, state, p)
    a in legal || throw(ArgumentError("Decision strategy produced illegal action $a for player $p."))
    return a
end

"""
Sample the current kernel action induced by a strategy profile at `(game, state)`.

This is the runtime bridge from:
- per-player strategies
- game state and information structure
to a kernel action usable by `Kernel.step`.
"""
function sample_joint_action(profile::StrategyProfiles.StrategyProfile,
                               game::Kernel.AbstractGame,
                               state,
                               rng::AbstractRNG = Random.default_rng())
    require_profile_length(profile, game)

    nk = Kernel.node_kind(game, state)

    if nk == Kernel.DECISION
        p = Kernel.only_acting_player(game, state)
        a = sample_strategy_action(profile[p], game, state, p, rng)
        return _check_legal_action(game, state, p, a)

    elseif nk == Kernel.SIMULTANEOUS
        aps = Tuple(Kernel.acting_players(game, state))
        acts = ntuple(i -> begin
            p = aps[i]
            a = sample_strategy_action(profile[p], game, state, p, rng)
            _check_legal_action(game, state, p, a)
        end, length(aps))
        ja = Kernel.JointAction(acts)
        return Kernel.validate_joint_action(game, state, ja)

    elseif nk == Kernel.CHANCE
        return Kernel.SampleChance()

    else
        throw(ArgumentError("Cannot sample an action at a terminal node."))
    end
end

function sample_joint_action(profile::Tuple,
                               game::Kernel.AbstractGame,
                               state,
                               rng::AbstractRNG = Random.default_rng())
    return sample_joint_action(
        StrategyProfiles.StrategyProfile(profile),
        game,
        state,
        rng,
    )
end

end