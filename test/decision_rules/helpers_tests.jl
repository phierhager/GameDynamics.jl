module TestHelpers

using Random
using Test

using GameLab
using GameLab.StrategyInterface
using GameLab.LocalStrategies

export DummyCustomContext
export DummyStatefulRule
export DummyFiniteStateRule
export NoLikelihoodNoContextRule
export NoLikelihoodObservationRule

struct DummyCustomContext <: StrategyInterface.AbstractContextKind end

"""
Simple unconditioned strategy with configurable metadata for profile tests.
"""
struct DummyStatefulRule{A} <:
       StrategyInterface.AbstractStrategy
    action::A
end

StrategyInterface.context_kind(::Type{<:DummyStatefulRule}) =
    StrategyInterface.NoContext()

StrategyInterface.internal_state_class(::Type{<:DummyStatefulRule{A,ISC}}) where {A,ISC} =
    ISC()

StrategyInterface.sample_action(strategy::DummyStatefulRule,
                                     rng::AbstractRNG = Random.default_rng()) = strategy.action

StrategyInterface.action_probability(strategy::DummyStatefulRule, action) =
    action == strategy.action ? 1.0 : 0.0

struct DummyFiniteStateRule{A} <: StrategyInterface.AbstractStrategy
    action::A
end

StrategyInterface.context_kind(::Type{<:DummyFiniteStateRule}) =
    StrategyInterface.NoContext()

StrategyInterface.internal_state_class(::Type{<:DummyFiniteStateRule}) =
    StrategyInterface.FiniteStateController()

StrategyInterface.sample_action(strategy::DummyFiniteStateRule,
                                     rng::AbstractRNG = Random.default_rng()) = strategy.action

StrategyInterface.action_probability(strategy::DummyFiniteStateRule, action) =
    action == strategy.action ? 1.0 : 0.0

"""
Contextual strategy without likelihood, to verify MethodError behavior.
"""
const NoLikelihoodNoContextRule =
    LocalStrategies.unconditioned_strategy(rng -> :a)

const NoLikelihoodObservationRule =
    LocalStrategies.observation_strategy((obs, rng) -> obs)

end