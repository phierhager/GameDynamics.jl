module ExtensiveForm

using Random
using ..Kernel
using ..Exact
using ..Capabilities
using ..Strategies

export infoset
export local_behavior
export behavior_action_probability
export sample_behavior_action
export sample_behavior_joint_action
export is_behavior_defined

function infoset(game::Kernel.AbstractGame, state, player::Int)
    if Capabilities.has_information_state(typeof(game)) === Val(true)
        return Exact.information_state(game, state, player)
    end
    return Kernel.observe(game, state, player)
end

local_behavior(strategy, game::Kernel.AbstractGame, state, player::Int) =
    Strategies.local_strategy(strategy, infoset(game, state, player))

behavior_action_probability(strategy, game::Kernel.AbstractGame, state, player::Int, action) =
    Strategies.probability(strategy, infoset(game, state, player), action)

sample_behavior_action(strategy, game::Kernel.AbstractGame, state, player::Int,
                       rng::AbstractRNG = Random.default_rng()) =
    Strategies.sample_action(strategy, infoset(game, state, player), rng)

function sample_behavior_joint_action(profile::Tuple{Vararg{Strategies.AbstractStrategy,N}},
                                      game::Kernel.AbstractGame,
                                      state,
                                      rng::AbstractRNG = Random.default_rng()) where {N}
    acts = ntuple(i -> sample_behavior_action(profile[i], game, state, i, rng), N)
    return Kernel.JointAction{N}(acts)
end

sample_behavior_joint_action(profile::Strategies.StrategyProfile{N},
                             game::Kernel.AbstractGame,
                             state,
                             rng::AbstractRNG = Random.default_rng()) where {N} =
    sample_behavior_joint_action(profile.strategies, game, state, rng)

function is_behavior_defined(strategy, game::Kernel.AbstractGame, state, player::Int)
    try
        local_behavior(strategy, game, state, player)
        return true
    catch
        return false
    end
end

end