module ContinuousDecisionRules

using Random
using ..DecisionRulesInterface

export SamplerDecisionRule
export SamplerDensityDecisionRule

"""
Sampler-only continuous decision rule.

Useful for rollout/sampling workflows.
"""
struct SamplerDecisionRule{S,Dom} <: DecisionRulesInterface.AbstractDecisionRule
    sampler::S
    domain::Dom
end

"""
Sampler + density continuous decision rule.

Useful when both simulation and density evaluation are needed.
"""
struct SamplerDensityDecisionRule{S,D,Dom} <: DecisionRulesInterface.AbstractDecisionRule
    sampler::S
    density_fn::D
    domain::Dom
end

DecisionRulesInterface.conditioning_kind(::Type{<:SamplerDecisionRule}) =
    DecisionRulesInterface.NoConditioning()
DecisionRulesInterface.conditioning_kind(::Type{<:SamplerDensityDecisionRule}) =
    DecisionRulesInterface.NoConditioning()

DecisionRulesInterface.memory_class(::Type{<:SamplerDecisionRule}) =
    DecisionRulesInterface.Memoryless()
DecisionRulesInterface.memory_class(::Type{<:SamplerDensityDecisionRule}) =
    DecisionRulesInterface.Memoryless()

DecisionRulesInterface.support(rule::SamplerDecisionRule) = rule.domain
DecisionRulesInterface.support(rule::SamplerDensityDecisionRule) = rule.domain

DecisionRulesInterface.sample_action(rule::SamplerDecisionRule,
                                     rng::AbstractRNG = Random.default_rng()) =
    rule.sampler(rng)

DecisionRulesInterface.sample_action(rule::SamplerDensityDecisionRule,
                                     rng::AbstractRNG = Random.default_rng()) =
    rule.sampler(rng)

DecisionRulesInterface.action_density(rule::SamplerDensityDecisionRule, action) =
    rule.density_fn(action)

end