module BehaviorDecisionRules

using Random
using ..DecisionRulesInterface

export CallableBehaviorRule
export TableBehaviorRule
export DenseBehaviorRule
export DenseVectorBehaviorRule

"""
Callable behavior rule.

`f(context)` must return a local decision rule.
"""
struct CallableBehaviorRule{F} <: DecisionRulesInterface.AbstractDecisionRule
    f::F
end

"""
Dictionary-backed behavior rule.

Good for convenience, not the fastest option.
"""
struct TableBehaviorRule{K,R,M<:AbstractDict{K,R}} <: DecisionRulesInterface.AbstractDecisionRule
    table::M
end

"""
Tuple-backed dense behavior rule for small fixed integer-indexed maps.
Assumes contexts are encoded as 1-based integers.
"""
struct DenseBehaviorRule{R,T<:Tuple{Vararg{R}}} <: DecisionRulesInterface.AbstractDecisionRule
    table::T
end

"""
Vector-backed dense behavior rule for larger integer-indexed maps.
Assumes contexts are encoded as 1-based integers.
"""
struct DenseVectorBehaviorRule{R,V<:AbstractVector{R}} <: DecisionRulesInterface.AbstractDecisionRule
    table::V
end

DecisionRulesInterface.conditioning_kind(::Type{<:CallableBehaviorRule}) =
    DecisionRulesInterface.InfosetConditioning()
DecisionRulesInterface.conditioning_kind(::Type{<:TableBehaviorRule}) =
    DecisionRulesInterface.InfosetConditioning()
DecisionRulesInterface.conditioning_kind(::Type{<:DenseBehaviorRule}) =
    DecisionRulesInterface.InfosetConditioning()
DecisionRulesInterface.conditioning_kind(::Type{<:DenseVectorBehaviorRule}) =
    DecisionRulesInterface.InfosetConditioning()

DecisionRulesInterface.memory_class(::Type{<:CallableBehaviorRule}) =
    DecisionRulesInterface.Memoryless()
DecisionRulesInterface.memory_class(::Type{<:TableBehaviorRule}) =
    DecisionRulesInterface.Memoryless()
DecisionRulesInterface.memory_class(::Type{<:DenseBehaviorRule}) =
    DecisionRulesInterface.Memoryless()
DecisionRulesInterface.memory_class(::Type{<:DenseVectorBehaviorRule}) =
    DecisionRulesInterface.Memoryless()

DecisionRulesInterface.local_rule(rule::CallableBehaviorRule, context) = rule.f(context)

function DecisionRulesInterface.local_rule(rule::TableBehaviorRule, context)
    haskey(rule.table, context) ||
        throw(KeyError("No local decision rule stored for context $context."))
    return rule.table[context]
end

DecisionRulesInterface.local_rule(rule::DenseBehaviorRule, context::Int) =
    rule.table[context]

DecisionRulesInterface.local_rule(rule::DenseVectorBehaviorRule, context::Int) =
    rule.table[context]

DecisionRulesInterface.sample_action(rule::CallableBehaviorRule,
                                     context,
                                     rng::AbstractRNG = Random.default_rng()) =
    DecisionRulesInterface.sample_action(
        DecisionRulesInterface.local_rule(rule, context), rng
    )

DecisionRulesInterface.sample_action(rule::TableBehaviorRule,
                                     context,
                                     rng::AbstractRNG = Random.default_rng()) =
    DecisionRulesInterface.sample_action(
        DecisionRulesInterface.local_rule(rule, context), rng
    )

DecisionRulesInterface.sample_action(rule::DenseBehaviorRule,
                                     context::Int,
                                     rng::AbstractRNG = Random.default_rng()) =
    DecisionRulesInterface.sample_action(
        DecisionRulesInterface.local_rule(rule, context), rng
    )

DecisionRulesInterface.sample_action(rule::DenseVectorBehaviorRule,
                                     context::Int,
                                     rng::AbstractRNG = Random.default_rng()) =
    DecisionRulesInterface.sample_action(
        DecisionRulesInterface.local_rule(rule, context), rng
    )

DecisionRulesInterface.action_probability(rule::CallableBehaviorRule, context, action) =
    DecisionRulesInterface.action_probability(
        DecisionRulesInterface.local_rule(rule, context), action
    )

DecisionRulesInterface.action_probability(rule::TableBehaviorRule, context, action) =
    DecisionRulesInterface.action_probability(
        DecisionRulesInterface.local_rule(rule, context), action
    )

DecisionRulesInterface.action_probability(rule::DenseBehaviorRule, context::Int, action) =
    DecisionRulesInterface.action_probability(
        DecisionRulesInterface.local_rule(rule, context), action
    )

DecisionRulesInterface.action_probability(rule::DenseVectorBehaviorRule, context::Int, action) =
    DecisionRulesInterface.action_probability(
        DecisionRulesInterface.local_rule(rule, context), action
    )

DecisionRulesInterface.action_density(rule::CallableBehaviorRule, context, action) =
    DecisionRulesInterface.action_density(
        DecisionRulesInterface.local_rule(rule, context), action
    )

DecisionRulesInterface.action_density(rule::TableBehaviorRule, context, action) =
    DecisionRulesInterface.action_density(
        DecisionRulesInterface.local_rule(rule, context), action
    )

DecisionRulesInterface.action_density(rule::DenseBehaviorRule, context::Int, action) =
    DecisionRulesInterface.action_density(
        DecisionRulesInterface.local_rule(rule, context), action
    )

DecisionRulesInterface.action_density(rule::DenseVectorBehaviorRule, context::Int, action) =
    DecisionRulesInterface.action_density(
        DecisionRulesInterface.local_rule(rule, context), action
    )

end