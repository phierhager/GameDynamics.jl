module StrategyProfiles

using Random
using ..StrategyInterface

export StrategyProfile
export num_strategies

export context_kinds
export internal_state_classes

export is_unconditioned
export is_observation_conditioned
export is_state_conditioned
export is_history_conditioned
export is_infoset_conditioned
export is_stateless

export require_unconditioned
export require_observation_conditioned
export require_state_conditioned
export require_history_conditioned
export require_infoset_conditioned
export require_stateless

export unconditioned_strategy_profile
export observation_strategy_profile
export state_strategy_profile
export history_strategy_profile
export infoset_strategy_profile

export sample_joint_action
export joint_action_probability
export joint_action_density

"""
Tuple-backed fixed-player strategy profile.

The field type is left as a concrete tuple type to preserve heterogeneous strategy
types for inference. Constructor validation enforces that all entries are
strategies.

Profiles are allowed to be heterogeneous in context kind. The homogeneous
predicate/require helpers below are convenience utilities, not a claim that
mixed-context profiles are invalid or unusual.
"""
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
Base.eltype(::Type{<:StrategyProfile}) = StrategyInterface.AbstractStrategy
Base.firstindex(::StrategyProfile) = 1
Base.lastindex(p::StrategyProfile) = length(p)
Base.Tuple(p::StrategyProfile) = p.strategies

num_strategies(p::StrategyProfile{N}) where {N} = N
num_strategies(p::Tuple) = length(p)

# ----------------------------------------------------------------------
# Introspection
# ----------------------------------------------------------------------

context_kinds(profile::StrategyProfile) =
    ntuple(i -> StrategyInterface.context_kind(profile[i]), length(profile))

internal_state_classes(profile::StrategyProfile) =
    ntuple(i -> StrategyInterface.internal_state_class(profile[i]), length(profile))

@inline function _all_context_kind(profile::StrategyProfile,
                                   ::Type{K}) where {K<:StrategyInterface.AbstractContextKind}
    @inbounds for i in eachindex(profile.strategies)
        StrategyInterface.context_kind(profile[i]) isa K || return false
    end
    return true
end

is_unconditioned(profile::StrategyProfile) =
    _all_context_kind(profile, StrategyInterface.NoContext)

is_observation_conditioned(profile::StrategyProfile) =
    _all_context_kind(profile, StrategyInterface.ObservationContext)

is_state_conditioned(profile::StrategyProfile) =
    _all_context_kind(profile, StrategyInterface.StateContext)

is_history_conditioned(profile::StrategyProfile) =
    _all_context_kind(profile, StrategyInterface.HistoryContext)

is_infoset_conditioned(profile::StrategyProfile) =
    _all_context_kind(profile, StrategyInterface.InfosetContext)

function is_stateless(profile::StrategyProfile)
    @inbounds for i in eachindex(profile.strategies)
        StrategyInterface.internal_state_class(profile[i]) isa StrategyInterface.Stateless || return false
    end
    return true
end

# ----------------------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------------------

function require_unconditioned(profile::StrategyProfile)
    is_unconditioned(profile) || throw(ArgumentError(
        "Expected an unconditioned strategy profile."
    ))
    return profile
end

function require_observation_conditioned(profile::StrategyProfile)
    is_observation_conditioned(profile) || throw(ArgumentError(
        "Expected an observation-conditioned strategy profile."
    ))
    return profile
end

function require_state_conditioned(profile::StrategyProfile)
    is_state_conditioned(profile) || throw(ArgumentError(
        "Expected a state-conditioned strategy profile."
    ))
    return profile
end

function require_history_conditioned(profile::StrategyProfile)
    is_history_conditioned(profile) || throw(ArgumentError(
        "Expected a history-conditioned strategy profile."
    ))
    return profile
end

function require_infoset_conditioned(profile::StrategyProfile)
    is_infoset_conditioned(profile) || throw(ArgumentError(
        "Expected an infoset-conditioned strategy profile."
    ))
    return profile
end

function require_stateless(profile::StrategyProfile)
    is_stateless(profile) || throw(ArgumentError(
        "Expected a stateless strategy profile."
    ))
    return profile
end

# ----------------------------------------------------------------------
# Convenience constructors
# ----------------------------------------------------------------------

unconditioned_strategy_profile(strategies::Tuple) =
    require_unconditioned(StrategyProfile(strategies))

observation_strategy_profile(strategies::Tuple) =
    require_observation_conditioned(StrategyProfile(strategies))

state_strategy_profile(strategies::Tuple) =
    require_state_conditioned(StrategyProfile(strategies))

history_strategy_profile(strategies::Tuple) =
    require_history_conditioned(StrategyProfile(strategies))

infoset_strategy_profile(strategies::Tuple) =
    require_infoset_conditioned(StrategyProfile(strategies))

# ----------------------------------------------------------------------
# Joint profile utilities
# ----------------------------------------------------------------------

@inline function _sample_joint_action_tuple(strategies::Tuple, rng::AbstractRNG)
    acts = ntuple(i -> StrategyInterface.sample_action(strategies[i], rng), length(strategies))
    return acts
end

"""
Sample a joint action tuple from an unconditioned profile.

This is only valid when all component strategies have context kind `NoContext`.
"""
function sample_joint_action(profile::StrategyProfile,
                             rng::AbstractRNG = Random.default_rng())
    require_unconditioned(profile)
    return _sample_joint_action_tuple(profile.strategies, rng)
end

function sample_joint_action(profile::Tuple{Vararg{StrategyInterface.AbstractStrategy,N}},
                             rng::AbstractRNG = Random.default_rng()) where {N}
    return sample_joint_action(StrategyProfile(profile), rng)
end

function sample_joint_action(profile::Tuple,
                             rng::AbstractRNG = Random.default_rng())
    return sample_joint_action(StrategyProfile(profile), rng)
end

@inline function _joint_action_probability_tuple(strategies::Tuple, actions::Tuple)
    length(strategies) == length(actions) ||
        throw(ArgumentError("Profile and action tuple must have the same length."))
    p = 1.0
    @inbounds for i in eachindex(strategies)
        p *= StrategyInterface.action_probability(strategies[i], actions[i])
    end
    return p
end

"""
Return the joint probability of an action tuple under an unconditioned profile.
"""
function joint_action_probability(profile::StrategyProfile, actions::Tuple)
    require_unconditioned(profile)
    return _joint_action_probability_tuple(profile.strategies, actions)
end

function joint_action_probability(profile::Tuple{Vararg{StrategyInterface.AbstractStrategy,N}},
                           actions::Tuple) where {N}
    return joint_action_probability(StrategyProfile(profile), actions)
end

function joint_action_probability(profile::Tuple, actions::Tuple)
    return joint_action_probability(StrategyProfile(profile), actions)
end

@inline function _joint_action_density_tuple(strategies::Tuple, actions::Tuple)
    length(strategies) == length(actions) ||
        throw(ArgumentError("Profile and action tuple must have the same length."))
    d = 1.0
    @inbounds for i in eachindex(strategies)
        d *= StrategyInterface.action_density(strategies[i], actions[i])
    end
    return d
end

"""
Return the joint density of an action tuple under an unconditioned profile.
"""
function joint_action_density(profile::StrategyProfile, actions::Tuple)
    require_unconditioned(profile)
    return _joint_action_density_tuple(profile.strategies, actions)
end

function joint_action_density(profile::Tuple{Vararg{StrategyInterface.AbstractStrategy,N}},
                       actions::Tuple) where {N}
    return joint_action_density(StrategyProfile(profile), actions)
end

function joint_action_density(profile::Tuple, actions::Tuple)
    return joint_action_density(StrategyProfile(profile), actions)
end

end