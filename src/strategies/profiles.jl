module StrategyProfiles

using Random
using ..StrategyInterface

export StrategyProfile
export num_strategies

export sample_joint_action
export joint_action_probability
export joint_action_density

struct StrategyProfile{N,S<:Tuple}
    strategies::S
end

function StrategyProfile(strategies::S) where {S<:Tuple}
    N = length(strategies)
    @inbounds for i in 1:N
        strategies[i] isa StrategyInterface.AbstractStrategy ||
            throw(ArgumentError(
                "All entries of a StrategyProfile must subtype AbstractStrategy. Entry $i has type $(typeof(strategies[i]))."
            ))
    end
    return StrategyProfile{N,S}(strategies)
end

Base.getindex(p::StrategyProfile, i::Int) = p.strategies[i]
Base.length(::StrategyProfile{N}) where {N} = N
Base.iterate(p::StrategyProfile, st...) = iterate(p.strategies, st...)
Base.firstindex(::StrategyProfile) = 1
Base.lastindex(p::StrategyProfile) = length(p)
Base.Tuple(p::StrategyProfile) = p.strategies

num_strategies(p::StrategyProfile{N}) where {N} = N
num_strategies(p::Tuple) = length(p)

# ----------------------------------------------------------------------
# Local joint profile utilities
# ----------------------------------------------------------------------

function sample_joint_action(profile::StrategyProfile,
                             rng::AbstractRNG = Random.default_rng())
    @inbounds for i in eachindex(profile.strategies)
        profile[i] isa StrategyInterface.AbstractLocalStrategy ||
            throw(ArgumentError("sample_joint_action(profile, rng) requires all component strategies to be local strategies."))
    end
    return ntuple(i -> StrategyInterface.sample_action(profile[i], rng), length(profile))
end

function joint_action_probability(profile::StrategyProfile, actions::Tuple)
    length(profile) == length(actions) ||
        throw(ArgumentError("Profile and action tuple must have the same length."))
    p = 1.0
    @inbounds for i in eachindex(profile.strategies)
        s = profile[i]
        s isa StrategyInterface.AbstractLocalStrategy ||
            throw(ArgumentError("joint_action_probability(profile, actions) requires all component strategies to be local strategies."))
        p *= StrategyInterface.action_probability(s, actions[i])
    end
    return p
end

function joint_action_density(profile::StrategyProfile, actions::Tuple)
    length(profile) == length(actions) ||
        throw(ArgumentError("Profile and action tuple must have the same length."))
    d = 1.0
    @inbounds for i in eachindex(profile.strategies)
        s = profile[i]
        s isa StrategyInterface.AbstractLocalStrategy ||
            throw(ArgumentError("joint_action_density(profile, actions) requires all component strategies to be local strategies."))
        d *= StrategyInterface.action_density(s, actions[i])
    end
    return d
end

# ----------------------------------------------------------------------
# Record-conditioned joint profile utilities
# ----------------------------------------------------------------------

function sample_joint_action(profile::StrategyProfile,
                             record,
                             rng::AbstractRNG = Random.default_rng())
    @inbounds for i in eachindex(profile.strategies)
        profile[i] isa StrategyInterface.AbstractRecordStrategy ||
            throw(ArgumentError("sample_joint_action(profile, record, rng) requires all component strategies to be record strategies."))
    end
    return ntuple(i -> StrategyInterface.sample_action(profile[i], record, rng), length(profile))
end

function joint_action_probability(profile::StrategyProfile, record, actions::Tuple)
    length(profile) == length(actions) ||
        throw(ArgumentError("Profile and action tuple must have the same length."))
    p = 1.0
    @inbounds for i in eachindex(profile.strategies)
        s = profile[i]
        s isa StrategyInterface.AbstractRecordStrategy ||
            throw(ArgumentError("joint_action_probability(profile, record, actions) requires all component strategies to be record strategies."))
        p *= StrategyInterface.action_probability(s, record, actions[i])
    end
    return p
end

function joint_action_density(profile::StrategyProfile, record, actions::Tuple)
    length(profile) == length(actions) ||
        throw(ArgumentError("Profile and action tuple must have the same length."))
    d = 1.0
    @inbounds for i in eachindex(profile.strategies)
        s = profile[i]
        s isa StrategyInterface.AbstractRecordStrategy ||
            throw(ArgumentError("joint_action_density(profile, record, actions) requires all component strategies to be record strategies."))
        d *= StrategyInterface.action_density(s, record, actions[i])
    end
    return d
end

end