module LookupDecisionRules

using Random
using ..DecisionRulesInterface

export CallableLookupRule
export TableLookupRule
export DenseLookupRule
export DenseVectorLookupRule

"""
Callable lookup rule.

`f(context)` must return a local decision rule.
"""
struct CallableLookupRule{
    CK<:DecisionRulesInterface.AbstractContextKind,
    F,
    ISC<:DecisionRulesInterface.AbstractInternalStateClass,
} <: DecisionRulesInterface.AbstractDecisionRule
    f::F
    internal_state::ISC
end

"""
Dictionary-backed lookup rule with explicit key and value types.
"""
struct TableLookupRule{
    CK<:DecisionRulesInterface.AbstractContextKind,
    K,
    R<:DecisionRulesInterface.AbstractDecisionRule,
    T<:AbstractDict{K,R},
    ISC<:DecisionRulesInterface.AbstractInternalStateClass,
} <: DecisionRulesInterface.AbstractDecisionRule
    table::T
    internal_state::ISC
end

"""
Tuple-backed dense lookup rule for small fixed integer-indexed maps.
Assumes contexts are encoded as 1-based integers.
"""
struct DenseLookupRule{
    CK<:DecisionRulesInterface.AbstractContextKind,
    R<:DecisionRulesInterface.AbstractDecisionRule,
    T<:Tuple{Vararg{R}},
    ISC<:DecisionRulesInterface.AbstractInternalStateClass,
} <: DecisionRulesInterface.AbstractDecisionRule
    table::T
    internal_state::ISC
end

"""
Vector-backed dense lookup rule for larger integer-indexed maps.
Assumes contexts are encoded as 1-based integers.
"""
struct DenseVectorLookupRule{
    CK<:DecisionRulesInterface.AbstractContextKind,
    R<:DecisionRulesInterface.AbstractDecisionRule,
    V<:AbstractVector{R},
    ISC<:DecisionRulesInterface.AbstractInternalStateClass,
} <: DecisionRulesInterface.AbstractDecisionRule
    table::V
    internal_state::ISC
end

CallableLookupRule(::Type{CK}, f;
                   internal_state::DecisionRulesInterface.AbstractInternalStateClass = DecisionRulesInterface.Stateless()) where {CK<:DecisionRulesInterface.AbstractContextKind} =
    CallableLookupRule{CK,typeof(f),typeof(internal_state)}(f, internal_state)

TableLookupRule(::Type{CK}, table::T;
                internal_state::DecisionRulesInterface.AbstractInternalStateClass = DecisionRulesInterface.Stateless()) where {CK<:DecisionRulesInterface.AbstractContextKind,K,R<:DecisionRulesInterface.AbstractDecisionRule,T<:AbstractDict{K,R}} =
    TableLookupRule{CK,K,R,T,typeof(internal_state)}(table, internal_state)

function DenseLookupRule(::Type{CK}, table::T;
                         internal_state::DecisionRulesInterface.AbstractInternalStateClass = DecisionRulesInterface.Stateless()) where {CK<:DecisionRulesInterface.AbstractContextKind,R,T<:Tuple{Vararg{R}}}
    @inbounds for i in eachindex(table)
        table[i] isa DecisionRulesInterface.AbstractDecisionRule ||
            throw(ArgumentError("DenseLookupRule entries must be decision rules. Entry $i has type $(typeof(table[i]))."))
    end
    return DenseLookupRule{CK,R,T,typeof(internal_state)}(table, internal_state)
end

function DenseVectorLookupRule(::Type{CK}, table::V;
                               internal_state::DecisionRulesInterface.AbstractInternalStateClass = DecisionRulesInterface.Stateless()) where {CK<:DecisionRulesInterface.AbstractContextKind,R,V<:AbstractVector{R}}
    @inbounds for i in eachindex(table)
        table[i] isa DecisionRulesInterface.AbstractDecisionRule ||
            throw(ArgumentError("DenseVectorLookupRule entries must be decision rules. Entry $i has type $(typeof(table[i]))."))
    end
    return DenseVectorLookupRule{CK,R,V,typeof(internal_state)}(table, internal_state)
end

DecisionRulesInterface.context_kind(::Type{<:CallableLookupRule{CK}}) where {CK} = CK()
DecisionRulesInterface.context_kind(::Type{<:TableLookupRule{CK}}) where {CK} = CK()
DecisionRulesInterface.context_kind(::Type{<:DenseLookupRule{CK}}) where {CK} = CK()
DecisionRulesInterface.context_kind(::Type{<:DenseVectorLookupRule{CK}}) where {CK} = CK()

DecisionRulesInterface.internal_state_class(::Type{<:CallableLookupRule{CK,F,ISC}}) where {CK,F,ISC<:DecisionRulesInterface.AbstractInternalStateClass} =
    ISC()
DecisionRulesInterface.internal_state_class(::Type{<:TableLookupRule{CK,K,R,T,ISC}}) where {CK,K,R,T,ISC<:DecisionRulesInterface.AbstractInternalStateClass} =
    ISC()
DecisionRulesInterface.internal_state_class(::Type{<:DenseLookupRule{CK,R,T,ISC}}) where {CK,R,T,ISC<:DecisionRulesInterface.AbstractInternalStateClass} =
    ISC()
DecisionRulesInterface.internal_state_class(::Type{<:DenseVectorLookupRule{CK,R,V,ISC}}) where {CK,R,V,ISC<:DecisionRulesInterface.AbstractInternalStateClass} =
    ISC()

function DecisionRulesInterface.local_rule(rule::CallableLookupRule, context)
    local_rule = rule.f(context)
    local_rule isa DecisionRulesInterface.AbstractDecisionRule ||
        throw(ArgumentError("CallableLookupRule must return an AbstractDecisionRule for context $context."))
    return local_rule
end

function DecisionRulesInterface.local_rule(rule::TableLookupRule, context)
    haskey(rule.table, context) ||
        throw(KeyError("No local decision rule stored for context $context."))
    return rule.table[context]
end

DecisionRulesInterface.local_rule(rule::DenseLookupRule, context::Int) =
    rule.table[context]

DecisionRulesInterface.local_rule(rule::DenseVectorLookupRule, context::Int) =
    rule.table[context]

DecisionRulesInterface.sample_action(rule::CallableLookupRule,
                                     context,
                                     rng::AbstractRNG = Random.default_rng()) =
    DecisionRulesInterface.sample_action(
        DecisionRulesInterface.local_rule(rule, context), rng
    )

DecisionRulesInterface.sample_action(rule::TableLookupRule,
                                     context,
                                     rng::AbstractRNG = Random.default_rng()) =
    DecisionRulesInterface.sample_action(
        DecisionRulesInterface.local_rule(rule, context), rng
    )

DecisionRulesInterface.sample_action(rule::DenseLookupRule,
                                     context::Int,
                                     rng::AbstractRNG = Random.default_rng()) =
    DecisionRulesInterface.sample_action(
        DecisionRulesInterface.local_rule(rule, context), rng
    )

DecisionRulesInterface.sample_action(rule::DenseVectorLookupRule,
                                     context::Int,
                                     rng::AbstractRNG = Random.default_rng()) =
    DecisionRulesInterface.sample_action(
        DecisionRulesInterface.local_rule(rule, context), rng
    )

DecisionRulesInterface.action_probability(rule::CallableLookupRule, context, action) =
    DecisionRulesInterface.action_probability(
        DecisionRulesInterface.local_rule(rule, context), action
    )

DecisionRulesInterface.action_probability(rule::TableLookupRule, context, action) =
    DecisionRulesInterface.action_probability(
        DecisionRulesInterface.local_rule(rule, context), action
    )

DecisionRulesInterface.action_probability(rule::DenseLookupRule, context::Int, action) =
    DecisionRulesInterface.action_probability(
        DecisionRulesInterface.local_rule(rule, context), action
    )

DecisionRulesInterface.action_probability(rule::DenseVectorLookupRule, context::Int, action) =
    DecisionRulesInterface.action_probability(
        DecisionRulesInterface.local_rule(rule, context), action
    )

DecisionRulesInterface.action_density(rule::CallableLookupRule, context, action) =
    DecisionRulesInterface.action_density(
        DecisionRulesInterface.local_rule(rule, context), action
    )

DecisionRulesInterface.action_density(rule::TableLookupRule, context, action) =
    DecisionRulesInterface.action_density(
        DecisionRulesInterface.local_rule(rule, context), action
    )

DecisionRulesInterface.action_density(rule::DenseLookupRule, context::Int, action) =
    DecisionRulesInterface.action_density(
        DecisionRulesInterface.local_rule(rule, context), action
    )

DecisionRulesInterface.action_density(rule::DenseVectorLookupRule, context::Int, action) =
    DecisionRulesInterface.action_density(
        DecisionRulesInterface.local_rule(rule, context), action
    )

end