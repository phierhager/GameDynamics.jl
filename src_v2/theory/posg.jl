module POSG

using Random
using ..Kernel
using ..Capabilities
using ..Spec
import ..Strategies
using ..Classification

export POSGProfile
export DecPOMDPProfile
export sample_joint_action
export local_observations

struct POSGProfile{P}
    policies::P
end

struct DecPOMDPProfile{P}
    policies::P
end

function sample_joint_action(profile::POSGProfile{P},
                             game::Kernel.AbstractGame,
                             state,
                             rng::AbstractRNG = Random.default_rng()) where {P}
    N = Kernel.num_players(game)
    acts = ntuple(i -> begin
        obs = Kernel.observe(game, state, i)
        Strategies.sample_action(profile.policies[i], obs, rng)
    end, N)
    return Kernel.JointAction{N}(acts)
end

function sample_joint_action(profile::DecPOMDPProfile{P},
                             game::Kernel.AbstractGame,
                             state,
                             rng::AbstractRNG = Random.default_rng()) where {P}
    N = Kernel.num_players(game)
    acts = ntuple(i -> begin
        obs = Kernel.observe(game, state, i)
        Strategies.sample_action(profile.policies[i], obs, rng)
    end, N)
    return Kernel.JointAction{N}(acts)
end

local_observations(game::Kernel.AbstractGame, state) =
    ntuple(i -> Kernel.observe(game, state, i), Kernel.num_players(game))

Classification.is_posg(game::Kernel.AbstractGame) =
    Spec.game_spec(game).stochastic &&
    (Capabilities.has_information_state(typeof(game)) === Val(true))

Classification.is_decpomdp(::DecPOMDPProfile) = true

end