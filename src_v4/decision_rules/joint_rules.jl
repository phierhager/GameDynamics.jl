module JointDecisionRules

using Random
using ..DecisionRulesInterface
using ..DecisionRuleInternalUtils

export CorrelatedActionRule

"""
Finite correlated action rule over joint action or recommendation tuples.

Intended semantics:
- support elements are joint tuples representing a current-stage recommendation
  or current-stage joint action, such as `(a1, a2, ..., aN)`

This object is not a local per-player decision rule. It is a joint behavioral
device over current-stage tuples.
"""
struct CorrelatedActionRule{S,P} <: DecisionRulesInterface.AbstractDecisionRule
    joint_tuples::S
    probs::P
end

function CorrelatedActionRule(joint_tuples, probs)
    tuples, p = DecisionRuleInternalUtils.canonicalize_joint_tuple_probs(joint_tuples, probs)
    return CorrelatedActionRule{typeof(tuples),typeof(p)}(tuples, p)
end

DecisionRulesInterface.context_kind(::Type{<:CorrelatedActionRule}) =
    DecisionRulesInterface.NoContext()

DecisionRulesInterface.internal_state_class(::Type{<:CorrelatedActionRule}) =
    DecisionRulesInterface.Stateless()

DecisionRulesInterface.support(rule::CorrelatedActionRule) = rule.joint_tuples
DecisionRulesInterface.probabilities(rule::CorrelatedActionRule) = rule.probs

function DecisionRulesInterface.sample_action(rule::CorrelatedActionRule,
                                              rng::AbstractRNG = Random.default_rng())
    r = rand(rng)
    c = 0.0
    S = DecisionRulesInterface.support(rule)
    P = DecisionRulesInterface.probabilities(rule)
    @inbounds for i in eachindex(P)
        c += P[i]
        if r <= c
            return S[i]
        end
    end
    return S[last(eachindex(P))]
end

function DecisionRulesInterface.action_probability(rule::CorrelatedActionRule, joint_tuple)
    S = DecisionRulesInterface.support(rule)
    P = DecisionRulesInterface.probabilities(rule)
    @inbounds for i in eachindex(P)
        if S[i] == joint_tuple
            return Float64(P[i])
        end
    end
    return 0.0
end

end