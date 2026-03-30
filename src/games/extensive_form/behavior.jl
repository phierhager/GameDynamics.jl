module ExtensiveFormBehavior

using Random

using ..Kernel
using ..StrategyInterface
using ..StrategyProfiles
using ..ExtensiveFormInfosets

export behavior_context
export sample_behavior_action
export behavior_action_probability
export sample_behavior_profile_action

@inline behavior_context(game::Kernel.AbstractGame, state, player::Int) =
    ExtensiveFormInfosets.infoset(game, state, player)

function sample_behavior_action(strategy::StrategyInterface.AbstractStrategy,
                                game::Kernel.AbstractGame,
                                state,
                                player::Int,
                                rng::AbstractRNG = Random.default_rng())
    ctx = behavior_context(game, state, player)

    if strategy isa StrategyInterface.AbstractLocalStrategy
        return StrategyInterface.sample_action(strategy, rng)
    elseif strategy isa StrategyInterface.AbstractRecordStrategy
        return StrategyInterface.sample_action(strategy, ctx, rng)
    else
        throw(ArgumentError("Unsupported strategy type $(typeof(strategy))."))
    end
end

function behavior_action_probability(strategy::StrategyInterface.AbstractStrategy,
                                     game::Kernel.AbstractGame,
                                     state,
                                     player::Int,
                                     action)
    ctx = behavior_context(game, state, player)

    if strategy isa StrategyInterface.AbstractLocalStrategy
        return StrategyInterface.action_probability(strategy, action)
    elseif strategy isa StrategyInterface.AbstractRecordStrategy
        return StrategyInterface.action_probability(strategy, ctx, action)
    else
        throw(ArgumentError("Unsupported strategy type $(typeof(strategy))."))
    end
end

function sample_behavior_profile_action(profile::StrategyProfiles.StrategyProfile,
                                        game::Kernel.AbstractGame,
                                        state,
                                        rng::AbstractRNG = Random.default_rng())
    Kernel.num_players(game) == length(profile) || throw(ArgumentError(
        "Profile length $(length(profile)) does not match number of players $(Kernel.num_players(game))."
    ))

    nk = Kernel.node_kind(game, state)

    if nk == Kernel.DECISION
        p = Kernel.only_acting_player(game, state)
        a = sample_behavior_action(profile[p], game, state, p, rng)
        legal = Kernel.legal_actions(game, state, p)
        a in legal || throw(ArgumentError(
            "Behavior profile produced illegal action $a for player $p."
        ))
        return a

    elseif nk == Kernel.SIMULTANEOUS
        aps = Tuple(Kernel.acting_players(game, state))
        acts = ntuple(i -> begin
            p = aps[i]
            a = sample_behavior_action(profile[p], game, state, p, rng)
            legal = Kernel.legal_actions(game, state, p)
            a in legal || throw(ArgumentError(
                "Behavior profile produced illegal action $a for player $p."
            ))
            a
        end, length(aps))

        ja = Kernel.JointAction(acts)
        return Kernel.validate_joint_action(game, state, ja)

    elseif nk == Kernel.CHANCE
        return Kernel.SampleChance()

    else
        throw(ArgumentError("Cannot sample an action at a terminal node."))
    end
end

function sample_behavior_profile_action(profile::Tuple,
                                        game::Kernel.AbstractGame,
                                        state,
                                        rng::AbstractRNG = Random.default_rng())
    return sample_behavior_profile_action(
        StrategyProfiles.StrategyProfile(profile),
        game,
        state,
        rng,
    )
end

end