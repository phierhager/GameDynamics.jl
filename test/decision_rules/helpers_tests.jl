module TestHelpers

using Random
using Test

using GameLab
using GameLab.DecisionRulesInterface
using GameLab.DirectDecisionRules

export DummyCustomContext
export DummyStatefulRule
export DummyFiniteStateRule
export NoLikelihoodNoContextRule
export NoLikelihoodObservationRule

struct DummyCustomContext <: DecisionRulesInterface.AbstractContextKind end

"""
Simple unconditioned rule with configurable metadata for profile tests.
"""
struct DummyStatefulRule{A,ISC<:DecisionRulesInterface.AbstractInternalStateClass} <:
       DecisionRulesInterface.AbstractDecisionRule
    action::A
    internal_state::ISC
end

DecisionRulesInterface.context_kind(::Type{<:DummyStatefulRule}) =
    DecisionRulesInterface.NoContext()

DecisionRulesInterface.internal_state_class(::Type{<:DummyStatefulRule{A,ISC}}) where {A,ISC} =
    ISC()

DecisionRulesInterface.sample_action(rule::DummyStatefulRule,
                                     rng::AbstractRNG = Random.default_rng()) = rule.action

DecisionRulesInterface.action_probability(rule::DummyStatefulRule, action) =
    action == rule.action ? 1.0 : 0.0

struct DummyFiniteStateRule{A} <: DecisionRulesInterface.AbstractDecisionRule
    action::A
end

DecisionRulesInterface.context_kind(::Type{<:DummyFiniteStateRule}) =
    DecisionRulesInterface.NoContext()

DecisionRulesInterface.internal_state_class(::Type{<:DummyFiniteStateRule}) =
    DecisionRulesInterface.FiniteStateController()

DecisionRulesInterface.sample_action(rule::DummyFiniteStateRule,
                                     rng::AbstractRNG = Random.default_rng()) = rule.action

DecisionRulesInterface.action_probability(rule::DummyFiniteStateRule, action) =
    action == rule.action ? 1.0 : 0.0

"""
Contextual rule without likelihood, to verify MethodError behavior.
"""
const NoLikelihoodNoContextRule =
    DirectDecisionRules.unconditioned_rule(rng -> :a)

const NoLikelihoodObservationRule =
    DirectDecisionRules.observation_rule((obs, rng) -> obs)

end