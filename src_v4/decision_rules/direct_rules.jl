module DirectDecisionRules

using Random
using ..DecisionRulesInterface
using ..DecisionRuleInternalUtils

export DeterministicDecisionRule
export FiniteMixedDecisionRule

export ContextualDecisionRule
export unconditioned_rule
export observation_rule
export state_rule
export history_rule
export infoset_rule
export custom_context_rule

export SamplerDecisionRule
export SamplerDensityDecisionRule

# ----------------------------------------------------------------------
# Deterministic / finite mixed
# ----------------------------------------------------------------------

struct DeterministicDecisionRule{A} <: DecisionRulesInterface.AbstractDecisionRule
    action::A
end

DecisionRulesInterface.context_kind(::Type{<:DeterministicDecisionRule}) =
    DecisionRulesInterface.NoContext()

DecisionRulesInterface.internal_state_class(::Type{<:DeterministicDecisionRule}) =
    DecisionRulesInterface.Stateless()

DecisionRulesInterface.support(rule::DeterministicDecisionRule) = (rule.action,)
DecisionRulesInterface.probabilities(::DeterministicDecisionRule) = (1.0,)

DecisionRulesInterface.sample_action(rule::DeterministicDecisionRule,
                                     rng::AbstractRNG = Random.default_rng()) =
    rule.action

DecisionRulesInterface.action_probability(rule::DeterministicDecisionRule, action) =
    action == rule.action ? 1.0 : 0.0

struct FiniteMixedDecisionRule{A,P} <: DecisionRulesInterface.AbstractDecisionRule
    actions::A
    probs::P
end

function FiniteMixedDecisionRule(actions, probs)
    acts, p = DecisionRuleInternalUtils.canonicalize_support_probs(actions, probs)
    return FiniteMixedDecisionRule{typeof(acts),typeof(p)}(acts, p)
end

FiniteMixedDecisionRule(probs) = FiniteMixedDecisionRule(Base.OneTo(length(probs)), probs)

FiniteMixedDecisionRule(actions::Tuple, probs::Tuple) =
    FiniteMixedDecisionRule(actions, probs)

DecisionRulesInterface.context_kind(::Type{<:FiniteMixedDecisionRule}) =
    DecisionRulesInterface.NoContext()

DecisionRulesInterface.internal_state_class(::Type{<:FiniteMixedDecisionRule}) =
    DecisionRulesInterface.Stateless()

DecisionRulesInterface.support(rule::FiniteMixedDecisionRule) = rule.actions
DecisionRulesInterface.probabilities(rule::FiniteMixedDecisionRule) = rule.probs

function DecisionRulesInterface.sample_action(rule::FiniteMixedDecisionRule,
                                              rng::AbstractRNG = Random.default_rng())
    r = rand(rng)
    c = 0.0
    A = DecisionRulesInterface.support(rule)
    P = DecisionRulesInterface.probabilities(rule)
    @inbounds for i in eachindex(P)
        c += P[i]
        if r <= c
            return A[i]
        end
    end
    return A[last(eachindex(P))]
end

function DecisionRulesInterface.action_probability(rule::FiniteMixedDecisionRule, action)
    A = DecisionRulesInterface.support(rule)
    P = DecisionRulesInterface.probabilities(rule)
    @inbounds for i in eachindex(P)
        if A[i] == action
            return Float64(P[i])
        end
    end
    return 0.0
end

# ----------------------------------------------------------------------
# Generic contextual direct rule
# ----------------------------------------------------------------------

"""
Generic callable decision rule parameterized by its context kind.

Expected signatures:
- `sampler(rng)` when `CK == NoContext`
- `sampler(context, rng)` otherwise

Optional likelihood signatures:
- `likelihood(action)` when `CK == NoContext`
- `likelihood(context, action)` otherwise

This type does not itself verify that the runtime-supplied context matches `CK`;
that compatibility check belongs to the runtime/query layer.
"""
struct ContextualDecisionRule{
    CK<:DecisionRulesInterface.AbstractContextKind,
    S,
    L,
    ISC<:DecisionRulesInterface.AbstractInternalStateClass,
} <: DecisionRulesInterface.AbstractDecisionRule
    sampler::S
    likelihood::L
    internal_state::ISC
end

function ContextualDecisionRule(::Type{CK},
                                sampler;
                                likelihood=nothing,
                                internal_state::DecisionRulesInterface.AbstractInternalStateClass=DecisionRulesInterface.Stateless()) where {CK<:DecisionRulesInterface.AbstractContextKind}
    return ContextualDecisionRule{CK,typeof(sampler),typeof(likelihood),typeof(internal_state)}(
        sampler, likelihood, internal_state
    )
end

DecisionRulesInterface.context_kind(::Type{<:ContextualDecisionRule{CK}}) where {CK} = CK()

DecisionRulesInterface.internal_state_class(::Type{<:ContextualDecisionRule{CK,S,L,ISC}}) where {CK,S,L,ISC<:DecisionRulesInterface.AbstractInternalStateClass} =
    ISC()

function DecisionRulesInterface.sample_action(rule::ContextualDecisionRule{DecisionRulesInterface.NoContext},
                                              rng::AbstractRNG = Random.default_rng())
    return rule.sampler(rng)
end

function DecisionRulesInterface.sample_action(rule::ContextualDecisionRule, context, rng::AbstractRNG = Random.default_rng())
    return rule.sampler(context, rng)
end

function DecisionRulesInterface.action_probability(rule::ContextualDecisionRule{DecisionRulesInterface.NoContext}, action)
    rule.likelihood === nothing &&
        throw(MethodError(DecisionRulesInterface.action_probability, (rule, action)))
    return rule.likelihood(action)
end

function DecisionRulesInterface.action_probability(rule::ContextualDecisionRule, context, action)
    rule.likelihood === nothing &&
        throw(MethodError(DecisionRulesInterface.action_probability, (rule, context, action)))
    return rule.likelihood(context, action)
end

# Convenience constructors

unconditioned_rule(sampler;
                   likelihood=nothing,
                   internal_state::DecisionRulesInterface.AbstractInternalStateClass=DecisionRulesInterface.Stateless()) =
    ContextualDecisionRule(DecisionRulesInterface.NoContext, sampler;
                           likelihood=likelihood,
                           internal_state=internal_state)

observation_rule(sampler;
                 likelihood=nothing,
                 internal_state::DecisionRulesInterface.AbstractInternalStateClass=DecisionRulesInterface.Stateless()) =
    ContextualDecisionRule(DecisionRulesInterface.ObservationContext, sampler;
                           likelihood=likelihood,
                           internal_state=internal_state)

state_rule(sampler;
           likelihood=nothing,
           internal_state::DecisionRulesInterface.AbstractInternalStateClass=DecisionRulesInterface.Stateless()) =
    ContextualDecisionRule(DecisionRulesInterface.StateContext, sampler;
                           likelihood=likelihood,
                           internal_state=internal_state)

history_rule(sampler;
             likelihood=nothing,
             internal_state::DecisionRulesInterface.AbstractInternalStateClass=DecisionRulesInterface.Stateless()) =
    ContextualDecisionRule(DecisionRulesInterface.HistoryContext, sampler;
                           likelihood=likelihood,
                           internal_state=internal_state)

infoset_rule(sampler;
             likelihood=nothing,
             internal_state::DecisionRulesInterface.AbstractInternalStateClass=DecisionRulesInterface.Stateless()) =
    ContextualDecisionRule(DecisionRulesInterface.InfosetContext, sampler;
                           likelihood=likelihood,
                           internal_state=internal_state)

custom_context_rule(sampler;
                    likelihood=nothing,
                    internal_state::DecisionRulesInterface.AbstractInternalStateClass=DecisionRulesInterface.Stateless()) =
    ContextualDecisionRule(DecisionRulesInterface.CustomContext, sampler;
                           likelihood=likelihood,
                           internal_state=internal_state)

# ----------------------------------------------------------------------
# Continuous sampler-based direct rules
# ----------------------------------------------------------------------

"""
Sampler-only decision rule over a continuous or otherwise non-enumerated action set.

`domain` denotes an admissible action set, not necessarily finite support in the
strict discrete sense.
"""
struct SamplerDecisionRule{S,Dom} <: DecisionRulesInterface.AbstractDecisionRule
    sampler::S
    domain::Dom
end

"""
Sampler + density decision rule over a continuous or otherwise non-enumerated
action set.

`domain` denotes an admissible action set, not necessarily finite support in the
strict discrete sense.
"""
struct SamplerDensityDecisionRule{S,D,Dom} <: DecisionRulesInterface.AbstractDecisionRule
    sampler::S
    density_fn::D
    domain::Dom
end

DecisionRulesInterface.context_kind(::Type{<:SamplerDecisionRule}) =
    DecisionRulesInterface.NoContext()
DecisionRulesInterface.context_kind(::Type{<:SamplerDensityDecisionRule}) =
    DecisionRulesInterface.NoContext()

DecisionRulesInterface.internal_state_class(::Type{<:SamplerDecisionRule}) =
    DecisionRulesInterface.Stateless()
DecisionRulesInterface.internal_state_class(::Type{<:SamplerDensityDecisionRule}) =
    DecisionRulesInterface.Stateless()

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