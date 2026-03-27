module ExtensiveForm

using Random
using ..Kernel
using ..Exact
using ..Strategies
using ..Interfaces
using ..Infosets

export local_behavior
export behavior_action_probability
export sample_behavior_action
export sample_behavior_profile_action
export is_behavior_defined

local_behavior(strategy, game::Kernel.AbstractGame, state, player::Int) =
    Strategies.local_strategy(strategy, Infosets.infoset(game, state, player))

behavior_action_probability(strategy, game::Kernel.AbstractGame, state, player::Int, action) =
    Strategies.probability(strategy, Infosets.infoset(game, state, player), action)

sample_behavior_action(strategy, game::Kernel.AbstractGame, state, player::Int,
                       rng::AbstractRNG = Random.default_rng()) =
    Strategies.sample_action(strategy, Infosets.infoset(game, state, player), rng)

function sample_behavior_profile_action(profile::Tuple{Vararg{Strategies.AbstractStrategy,N}},
                                        game::Kernel.AbstractGame,
                                        state,
                                        rng::AbstractRNG = Random.default_rng()) where {N}
    Kernel.num_players(game) == N || throw(ArgumentError(
        "Profile size $N does not match number of players $(Kernel.num_players(game))."
    ))

    nk = Kernel.node_kind(game, state)

    if nk == Kernel.DECISION
        p = Kernel.only_acting_player(game, state)
        a = sample_behavior_action(profile[p], game, state, p, rng)
        legal = Kernel.legal_actions(game, state, p)
        a in legal || throw(ArgumentError(
            "Behavior profile produced illegal action $a for decision player $p."
        ))
        return a

    elseif nk == Kernel.SIMULTANEOUS
        aps = Tuple(Kernel.acting_players(game, state))
        acts = ntuple(i -> begin
            p = aps[i]
            a = sample_behavior_action(profile[p], game, state, p, rng)
            legal = Kernel.legal_actions(game, state, p)
            a in legal || throw(ArgumentError(
                "Behavior profile produced illegal action $a for active player $p at local slot $i."
            ))
            a
        end, length(aps))

        ja = Kernel.joint_action(acts)
        return Kernel.validate_joint_action(game, state, ja)

    elseif nk == Kernel.CHANCE
        return Kernel.SampleChance()

    else
        throw(ArgumentError("Cannot sample a behavior-profile action at a terminal node."))
    end
end

sample_behavior_profile_action(profile::Strategies.StrategyProfile{N},
                               game::Kernel.AbstractGame,
                               state,
                               rng::AbstractRNG = Random.default_rng()) where {N} =
    sample_behavior_profile_action(profile.strategies, game, state, rng)

function is_behavior_defined(strategy, game::Kernel.AbstractGame, state, player::Int)
    try
        local_behavior(strategy, game, state, player)
        return true
    catch
        return false
    end
end

end