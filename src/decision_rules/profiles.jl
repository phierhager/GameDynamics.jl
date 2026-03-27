module DecisionRuleProfiles

using Random
using ..DecisionRulesInterface

export DecisionRuleProfile
export num_rules

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

export unconditioned_rule_profile
export observation_rule_profile
export state_rule_profile
export history_rule_profile
export infoset_rule_profile

export sample_joint_action
export joint_probability
export joint_density

"""
Tuple-backed fixed-player decision-rule profile.

The field type is left as a concrete tuple type to preserve heterogeneous rule
types for inference. Constructor validation enforces that all entries are
decision rules.

Profiles are allowed to be heterogeneous in context kind. The homogeneous
predicate/require helpers below are convenience utilities, not a claim that
mixed-context profiles are invalid or unusual.
"""
struct DecisionRuleProfile{N,R<:Tuple}
    rules::R
end

function DecisionRuleProfile(rules::R) where {R<:Tuple}
    N = length(rules)
    @inbounds for i in 1:N
        rules[i] isa DecisionRulesInterface.AbstractDecisionRule ||
            throw(ArgumentError(
                "All entries of a DecisionRuleProfile must subtype AbstractDecisionRule. Entry $i has type $(typeof(rules[i]))."
            ))
    end
    return DecisionRuleProfile{N,R}(rules)
end

Base.getindex(p::DecisionRuleProfile, i::Int) = p.rules[i]
Base.length(::DecisionRuleProfile{N}) where {N} = N
Base.iterate(p::DecisionRuleProfile, st...) = iterate(p.rules, st...)
Base.eltype(::Type{<:DecisionRuleProfile}) = DecisionRulesInterface.AbstractDecisionRule
Base.firstindex(::DecisionRuleProfile) = 1
Base.lastindex(p::DecisionRuleProfile) = length(p)
Base.Tuple(p::DecisionRuleProfile) = p.rules

num_rules(p::DecisionRuleProfile{N}) where {N} = N
num_rules(p::Tuple) = length(p)

# ----------------------------------------------------------------------
# Introspection
# ----------------------------------------------------------------------

context_kinds(profile::DecisionRuleProfile) =
    ntuple(i -> DecisionRulesInterface.context_kind(profile[i]), length(profile))

internal_state_classes(profile::DecisionRuleProfile) =
    ntuple(i -> DecisionRulesInterface.internal_state_class(profile[i]), length(profile))

@inline function _all_context_kind(profile::DecisionRuleProfile,
                                   ::Type{K}) where {K<:DecisionRulesInterface.AbstractContextKind}
    @inbounds for i in eachindex(profile.rules)
        DecisionRulesInterface.context_kind(profile[i]) isa K || return false
    end
    return true
end

is_unconditioned(profile::DecisionRuleProfile) =
    _all_context_kind(profile, DecisionRulesInterface.NoContext)

is_observation_conditioned(profile::DecisionRuleProfile) =
    _all_context_kind(profile, DecisionRulesInterface.ObservationContext)

is_state_conditioned(profile::DecisionRuleProfile) =
    _all_context_kind(profile, DecisionRulesInterface.StateContext)

is_history_conditioned(profile::DecisionRuleProfile) =
    _all_context_kind(profile, DecisionRulesInterface.HistoryContext)

is_infoset_conditioned(profile::DecisionRuleProfile) =
    _all_context_kind(profile, DecisionRulesInterface.InfosetContext)

function is_stateless(profile::DecisionRuleProfile)
    @inbounds for i in eachindex(profile.rules)
        DecisionRulesInterface.internal_state_class(profile[i]) isa DecisionRulesInterface.Stateless || return false
    end
    return true
end

# ----------------------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------------------

function require_unconditioned(profile::DecisionRuleProfile)
    is_unconditioned(profile) || throw(ArgumentError(
        "Expected an unconditioned decision-rule profile."
    ))
    return profile
end

function require_observation_conditioned(profile::DecisionRuleProfile)
    is_observation_conditioned(profile) || throw(ArgumentError(
        "Expected an observation-conditioned decision-rule profile."
    ))
    return profile
end

function require_state_conditioned(profile::DecisionRuleProfile)
    is_state_conditioned(profile) || throw(ArgumentError(
        "Expected a state-conditioned decision-rule profile."
    ))
    return profile
end

function require_history_conditioned(profile::DecisionRuleProfile)
    is_history_conditioned(profile) || throw(ArgumentError(
        "Expected a history-conditioned decision-rule profile."
    ))
    return profile
end

function require_infoset_conditioned(profile::DecisionRuleProfile)
    is_infoset_conditioned(profile) || throw(ArgumentError(
        "Expected an infoset-conditioned decision-rule profile."
    ))
    return profile
end

function require_stateless(profile::DecisionRuleProfile)
    is_stateless(profile) || throw(ArgumentError(
        "Expected a stateless decision-rule profile."
    ))
    return profile
end

# ----------------------------------------------------------------------
# Convenience constructors
# ----------------------------------------------------------------------

unconditioned_rule_profile(rules::Tuple) =
    require_unconditioned(DecisionRuleProfile(rules))

observation_rule_profile(rules::Tuple) =
    require_observation_conditioned(DecisionRuleProfile(rules))

state_rule_profile(rules::Tuple) =
    require_state_conditioned(DecisionRuleProfile(rules))

history_rule_profile(rules::Tuple) =
    require_history_conditioned(DecisionRuleProfile(rules))

infoset_rule_profile(rules::Tuple) =
    require_infoset_conditioned(DecisionRuleProfile(rules))

# ----------------------------------------------------------------------
# Joint profile utilities
# ----------------------------------------------------------------------

@inline function _sample_joint_action_tuple(rules::Tuple, rng::AbstractRNG)
    acts = ntuple(i -> DecisionRulesInterface.sample_action(rules[i], rng), length(rules))
    return acts
end

"""
Sample a joint action tuple from an unconditioned profile.

This is only valid when all local rules are `NoContext` rules.
"""
function sample_joint_action(profile::DecisionRuleProfile,
                             rng::AbstractRNG = Random.default_rng())
    require_unconditioned(profile)
    return _sample_joint_action_tuple(profile.rules, rng)
end

function sample_joint_action(profile::Tuple{Vararg{DecisionRulesInterface.AbstractDecisionRule,N}},
                             rng::AbstractRNG = Random.default_rng()) where {N}
    return sample_joint_action(DecisionRuleProfile(profile), rng)
end

function sample_joint_action(profile::Tuple,
                             rng::AbstractRNG = Random.default_rng())
    return sample_joint_action(DecisionRuleProfile(profile), rng)
end

@inline function _joint_probability_tuple(rules::Tuple, actions::Tuple)
    length(rules) == length(actions) ||
        throw(ArgumentError("Profile and action tuple must have the same length."))
    p = 1.0
    @inbounds for i in eachindex(rules)
        p *= DecisionRulesInterface.action_probability(rules[i], actions[i])
    end
    return p
end

"""
Return the joint probability of an action tuple under an unconditioned profile.
"""
function joint_probability(profile::DecisionRuleProfile, actions::Tuple)
    require_unconditioned(profile)
    return _joint_probability_tuple(profile.rules, actions)
end

function joint_probability(profile::Tuple{Vararg{DecisionRulesInterface.AbstractDecisionRule,N}},
                           actions::Tuple) where {N}
    return joint_probability(DecisionRuleProfile(profile), actions)
end

function joint_probability(profile::Tuple, actions::Tuple)
    return joint_probability(DecisionRuleProfile(profile), actions)
end

@inline function _joint_density_tuple(rules::Tuple, actions::Tuple)
    length(rules) == length(actions) ||
        throw(ArgumentError("Profile and action tuple must have the same length."))
    d = 1.0
    @inbounds for i in eachindex(rules)
        d *= DecisionRulesInterface.action_density(rules[i], actions[i])
    end
    return d
end

"""
Return the joint density of an action tuple under an unconditioned profile.
"""
function joint_density(profile::DecisionRuleProfile, actions::Tuple)
    require_unconditioned(profile)
    return _joint_density_tuple(profile.rules, actions)
end

function joint_density(profile::Tuple{Vararg{DecisionRulesInterface.AbstractDecisionRule,N}},
                       actions::Tuple) where {N}
    return joint_density(DecisionRuleProfile(profile), actions)
end

function joint_density(profile::Tuple, actions::Tuple)
    return joint_density(DecisionRuleProfile(profile), actions)
end

end