module ExtensiveForm

using Random
using ..Kernel
using ..Exact
using ..Strategies

export infoset
export local_behavior
export behavior_action_probability
export sample_behavior_action
export sample_behavior_profile_action
export is_behavior_defined

"""
Default infoset proxy.

If `Exact.information_state` is not specialized for the concrete game/state/player
combination, fall back to `Kernel.observe`.
"""
function infoset(game::Kernel.AbstractGame, state, player::Int)
    if applicable(Exact.information_state, game, state, player)
        return Exact.information_state(game, state, player)
    else
        return Kernel.observe(game, state, player)
    end
end

local_behavior(strategy, game::Kernel.AbstractGame, state, player::Int) =
    Strategies.local_strategy(strategy, infoset(game, state, player))

behavior_action_probability(strategy, game::Kernel.AbstractGame, state, player::Int, action) =
    Strategies.probability(strategy, infoset(game, state, player), action)
    
sample_behavior_action(strategy, game::Kernel.AbstractGame, state, player::Int,
                       rng::AbstractRNG = Random.default_rng()) =
    Strategies.sample_action(strategy, infoset(game, state, player), rng)

function sample_behavior_profile_action(profile::Tuple{Vararg{Strategies.AbstractStrategy,N}},
                                      game::Kernel.AbstractGame,
                                      state,
                                      rng::AbstractRNG = Random.default_rng()) where {N}
    acts = ntuple(i -> sample_behavior_action(profile[i], game, state, i, rng), N)
    return Kernel.joint_action(acts)
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