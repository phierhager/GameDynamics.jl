module BuiltinDecisionRules

using Random
using ..DecisionRulesInterface

export AbstractBuiltinDecisionRule
export UnconditionedDecisionRule
export ObservationDecisionRule
export StateDecisionRule

"""
Shared root for built-in concrete decision-rule types.
"""
abstract type AbstractBuiltinDecisionRule <: DecisionRulesInterface.AbstractDecisionRule end

"""
Built-in decision rule with no conditioning input at query time.

This means the rule is queried as `sample_action(rule, rng)`, not that the world
lacks state.
"""
struct UnconditionedDecisionRule{S,L,M<:DecisionRulesInterface.AbstractMemoryClass} <: AbstractBuiltinDecisionRule
    sampler::S
    likelihood::L
    memory::M
end

"""
Built-in decision rule queried on the current observation.
"""
struct ObservationDecisionRule{S,L,M<:DecisionRulesInterface.AbstractMemoryClass} <: AbstractBuiltinDecisionRule
    sampler::S
    likelihood::L
    memory::M
end

"""
Built-in decision rule queried on the true current underlying state.
"""
struct StateDecisionRule{S,L,M<:DecisionRulesInterface.AbstractMemoryClass} <: AbstractBuiltinDecisionRule
    sampler::S
    likelihood::L
    memory::M
end

# ----------------------------------------------------------------------
# Constructors
# ----------------------------------------------------------------------

"""
Create an unconditioned decision rule.

Expected sampler signature:
- `sampler(rng)`

Optional likelihood signature:
- `likelihood(action)`
"""
UnconditionedDecisionRule(sampler;
                          likelihood=nothing,
                          memory::DecisionRulesInterface.AbstractMemoryClass=DecisionRulesInterface.Memoryless()) =
    UnconditionedDecisionRule{typeof(sampler),typeof(likelihood),typeof(memory)}(
        sampler, likelihood, memory
    )

"""
Create an observation-conditioned decision rule.

Expected sampler signature:
- `sampler(observation, rng)`

Optional likelihood signature:
- `likelihood(observation, action)`
"""
ObservationDecisionRule(sampler;
                        likelihood=nothing,
                        memory::DecisionRulesInterface.AbstractMemoryClass=DecisionRulesInterface.Memoryless()) =
    ObservationDecisionRule{typeof(sampler),typeof(likelihood),typeof(memory)}(
        sampler, likelihood, memory
    )

"""
Create a state-conditioned decision rule.

Expected sampler signature:
- `sampler(state, rng)`

Optional likelihood signature:
- `likelihood(state, action)`
"""
StateDecisionRule(sampler;
                  likelihood=nothing,
                  memory::DecisionRulesInterface.AbstractMemoryClass=DecisionRulesInterface.Memoryless()) =
    StateDecisionRule{typeof(sampler),typeof(likelihood),typeof(memory)}(
        sampler, likelihood, memory
    )

# ----------------------------------------------------------------------
# Trait declarations
# ----------------------------------------------------------------------

DecisionRulesInterface.conditioning_kind(::Type{<:UnconditionedDecisionRule}) = DecisionRulesInterface.NoConditioning()
DecisionRulesInterface.conditioning_kind(::Type{<:ObservationDecisionRule}) = DecisionRulesInterface.ObservationConditioning()
DecisionRulesInterface.conditioning_kind(::Type{<:StateDecisionRule}) = DecisionRulesInterface.StateConditioning()

DecisionRulesInterface.memory_class(::Type{<:UnconditionedDecisionRule{S,L,M}}) where {S,L,M<:DecisionRulesInterface.AbstractMemoryClass} = M()
DecisionRulesInterface.memory_class(::Type{<:ObservationDecisionRule{S,L,M}}) where {S,L,M<:DecisionRulesInterface.AbstractMemoryClass} = M()
DecisionRulesInterface.memory_class(::Type{<:StateDecisionRule{S,L,M}}) where {S,L,M<:DecisionRulesInterface.AbstractMemoryClass} = M()

DecisionRulesInterface.has_action_likelihood(::Type{<:UnconditionedDecisionRule{S,L}}) where {S,L} = !(L <: Nothing)
DecisionRulesInterface.has_action_likelihood(::Type{<:ObservationDecisionRule{S,L}}) where {S,L} = !(L <: Nothing)
DecisionRulesInterface.has_action_likelihood(::Type{<:StateDecisionRule{S,L}}) where {S,L} = !(L <: Nothing)

# ----------------------------------------------------------------------
# Canonical query implementations
# ----------------------------------------------------------------------

DecisionRulesInterface.sample_action(rule::UnconditionedDecisionRule,
                                     rng::AbstractRNG = Random.default_rng()) =
    rule.sampler(rng)

DecisionRulesInterface.sample_action(rule::ObservationDecisionRule,
                                     observation,
                                     rng::AbstractRNG = Random.default_rng()) =
    rule.sampler(observation, rng)

DecisionRulesInterface.sample_action(rule::StateDecisionRule,
                                     state,
                                     rng::AbstractRNG = Random.default_rng()) =
    rule.sampler(state, rng)

function DecisionRulesInterface.action_likelihood(rule::UnconditionedDecisionRule, action)
    DecisionRulesInterface.has_action_likelihood(rule) ||
        throw(MethodError(DecisionRulesInterface.action_likelihood, (rule, action)))
    return rule.likelihood(action)
end

function DecisionRulesInterface.action_likelihood(rule::ObservationDecisionRule, observation, action)
    DecisionRulesInterface.has_action_likelihood(rule) ||
        throw(MethodError(DecisionRulesInterface.action_likelihood, (rule, observation, action)))
    return rule.likelihood(observation, action)
end

function DecisionRulesInterface.action_likelihood(rule::StateDecisionRule, state, action)
    DecisionRulesInterface.has_action_likelihood(rule) ||
        throw(MethodError(DecisionRulesInterface.action_likelihood, (rule, state, action)))
    return rule.likelihood(state, action)
end

end