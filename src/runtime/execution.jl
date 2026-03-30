module RuntimeStrategyExecution

using Random

using ..Kernel
using ..StrategyInterface
using ..StrategyProfiles

export require_profile_length
export player_record
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

function require_profile_length(profile::Tuple,
                                game::Kernel.AbstractGame)
    return require_profile_length(StrategyProfiles.StrategyProfile(profile), game)
end

"""
Construct the runtime query object passed to a record-conditioned strategy.

Default behavior:
- record strategies see the player's current observation via
  `Kernel.observe(game, state, player)`

This is intentionally a lightweight extensibility hook. If a specific strategy
family should instead consume infosets, full states, or another game-specific
query object, specialize `player_record(strategy, game, state, player)` for
that strategy type.
"""
function player_record(strategy::StrategyInterface.AbstractRecordStrategy,
                       game::Kernel.AbstractGame,
                       state,
                       player::Int)
    return Kernel.observe(game, state, player)
end

"""
Sample the action induced by a local strategy for `player` at `(game, state)`.
"""
function sample_strategy_action(strategy::StrategyInterface.AbstractLocalStrategy,
                                game::Kernel.AbstractGame,
                                state,
                                player::Int,
                                rng::AbstractRNG = Random.default_rng())
    return StrategyInterface.sample_action(strategy, rng)
end

"""
Sample the action induced by a record-conditioned strategy for `player` at
`(game, state)`.
"""
function sample_strategy_action(strategy::StrategyInterface.AbstractRecordStrategy,
                                game::Kernel.AbstractGame,
                                state,
                                player::Int,
                                rng::AbstractRNG = Random.default_rng())
    rec = player_record(strategy, game, state, player)
    return StrategyInterface.sample_action(strategy, rec, rng)
end

"""
Return the action probability induced by a local strategy for `player` at
`(game, state)`.
"""
function action_probability_at(strategy::StrategyInterface.AbstractLocalStrategy,
                               game::Kernel.AbstractGame,
                               state,
                               player::Int,
                               action)
    return StrategyInterface.action_probability(strategy, action)
end

"""
Return the action probability induced by a record-conditioned strategy for
`player` at `(game, state)`.
"""
function action_probability_at(strategy::StrategyInterface.AbstractRecordStrategy,
                               game::Kernel.AbstractGame,
                               state,
                               player::Int,
                               action)
    rec = player_record(strategy, game, state, player)
    return StrategyInterface.action_probability(strategy, rec, action)
end

@inline function _check_legal_action(game, state, p::Int, a)
    legal = Kernel.legal_actions(game, state, p)
    a in legal || throw(ArgumentError(
        "Strategy produced illegal action $a for player $p."
    ))
    return a
end

"""
Sample the current kernel action induced by a strategy profile at `(game, state)`.

This is the runtime bridge from:
- per-player strategies
- game state and observation structure
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