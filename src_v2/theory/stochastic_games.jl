module StochasticGames

using Random
using ..Kernel
using ..Strategies

export StationaryPolicyProfile
export OpenLoopPolicyProfile
export ObservationPolicyProfile
export sample_profile_action
export rollout_value

struct OpenLoopPolicyProfile{P<:Tuple}
    policies::P
end

function OpenLoopPolicyProfile(policies::P) where {P<:Tuple}
    @inbounds for i in eachindex(policies)
        policies[i] isa Strategies.AbstractStrategy ||
            throw(ArgumentError("All policies in an OpenLoopPolicyProfile must subtype AbstractStrategy."))
    end
    return OpenLoopPolicyProfile{P}(policies)
end

struct ObservationPolicyProfile{P<:Tuple}
    policies::P
end

function ObservationPolicyProfile(policies::P) where {P<:Tuple}
    @inbounds for i in eachindex(policies)
        policies[i] isa Strategies.AbstractStrategy ||
            throw(ArgumentError("All policies in an ObservationPolicyProfile must subtype AbstractStrategy."))
    end
    return ObservationPolicyProfile{P}(policies)
end

StationaryPolicyProfile(policies::Tuple) = OpenLoopPolicyProfile(policies)
StationaryPolicyProfile(policies::Strategies.StrategyProfile) = OpenLoopPolicyProfile(policies.strategies)

@inline _local_observations(game::Kernel.AbstractGame, state, N::Int) =
    ntuple(i -> Kernel.observe(game, state, i), N)

function sample_profile_action(profile::OpenLoopPolicyProfile,
                               game::Kernel.AbstractGame,
                               state,
                               rng::AbstractRNG = Random.default_rng())
    nk = Kernel.node_kind(game, state)

    if nk == Kernel.DECISION
        p = Kernel.current_player(game, state)
        return Strategies.sample_action(profile.policies[p], rng)
    elseif nk == Kernel.SIMULTANEOUS
        return Strategies.sample_joint_action(profile.policies, rng)
    elseif nk == Kernel.CHANCE
        return Kernel.SampleChance()
    else
        throw(ArgumentError("Cannot sample an action at a terminal node."))
    end
end

function sample_profile_action(profile::ObservationPolicyProfile,
                               game::Kernel.AbstractGame,
                               state,
                               rng::AbstractRNG = Random.default_rng())
    nk = Kernel.node_kind(game, state)

    if nk == Kernel.DECISION
        p = Kernel.current_player(game, state)
        obs = Kernel.observe(game, state, p)
        return Strategies.sample_action(profile.policies[p], obs, rng)
    elseif nk == Kernel.SIMULTANEOUS
        N = Kernel.num_players(game)
        obs = _local_observations(game, state, N)
        acts = ntuple(i -> Strategies.sample_action(profile.policies[i], obs[i], rng), N)
        return Kernel.JointAction{N}(acts)
    elseif nk == Kernel.CHANCE
        return Kernel.SampleChance()
    else
        throw(ArgumentError("Cannot sample an action at a terminal node."))
    end
end

function rollout_value(game::Kernel.AbstractGame,
                       profile;
                       rng::AbstractRNG = Random.default_rng(),
                       max_steps::Int = 1024,
                       discount::Float64 = 1.0)
    state = Kernel.init_state(game, rng)
    N = Kernel.num_players(game)
    totals = ntuple(_ -> 0.0, N)
    coeff = 1.0
    steps = 0

    while !Kernel.is_terminal(game, state) && steps < max_steps
        action = sample_profile_action(profile, game, state, rng)
        state, rewards, terminated = Kernel.step(game, state, action, rng)
        totals = ntuple(i -> totals[i] + coeff * rewards[i], N)
        coeff *= discount
        steps += 1
        terminated && break
    end

    return totals
end

end