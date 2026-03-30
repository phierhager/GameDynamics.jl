module LocalStrategies

using Random
using ..StrategyInterface
using ..StrategyInternalUtils

export DeterministicStrategy
export FiniteMixedStrategy
export SamplerStrategy
export SamplerDensityStrategy

# ----------------------------------------------------------------------
# Deterministic / finite mixed
# ----------------------------------------------------------------------

struct DeterministicStrategy{A} <: StrategyInterface.AbstractLocalStrategy
    action::A
end

StrategyInterface.support(strategy::DeterministicStrategy) = (strategy.action,)
StrategyInterface.probabilities(::DeterministicStrategy) = (1.0,)

StrategyInterface.sample_action(strategy::DeterministicStrategy,
                                rng::AbstractRNG = Random.default_rng()) =
    strategy.action

StrategyInterface.action_probability(strategy::DeterministicStrategy, action) =
    action == strategy.action ? 1.0 : 0.0

struct FiniteMixedStrategy{A,P} <: StrategyInterface.AbstractLocalStrategy
    actions::A
    probs::P

    function FiniteMixedStrategy(actions, probs)
        acts, p = StrategyInternalUtils.canonicalize_support_probs(actions, probs)
        return new{typeof(acts),typeof(p)}(acts, p)
    end
end

FiniteMixedStrategy(probs) =
    FiniteMixedStrategy(Base.OneTo(length(probs)), probs)

StrategyInterface.support(strategy::FiniteMixedStrategy) = strategy.actions
StrategyInterface.probabilities(strategy::FiniteMixedStrategy) = strategy.probs

function StrategyInterface.sample_action(strategy::FiniteMixedStrategy,
                                         rng::AbstractRNG = Random.default_rng())
    r = rand(rng)
    c = 0.0
    A = StrategyInterface.support(strategy)
    P = StrategyInterface.probabilities(strategy)
    @inbounds for i in eachindex(P)
        c += P[i]
        if r <= c
            return A[i]
        end
    end
    return A[last(eachindex(P))]
end

function StrategyInterface.action_probability(strategy::FiniteMixedStrategy, action)
    A = StrategyInterface.support(strategy)
    P = StrategyInterface.probabilities(strategy)
    @inbounds for i in eachindex(P)
        if A[i] == action
            return Float64(P[i])
        end
    end
    return 0.0
end

# ----------------------------------------------------------------------
# Continuous sampler-based local strategies
# ----------------------------------------------------------------------

struct SamplerStrategy{S,Dom} <: StrategyInterface.AbstractLocalStrategy
    sampler::S
    domain::Dom
end

struct SamplerDensityStrategy{S,D,Dom} <: StrategyInterface.AbstractLocalStrategy
    sampler::S
    density_fn::D
    domain::Dom
end

StrategyInterface.support(strategy::SamplerStrategy) = strategy.domain
StrategyInterface.support(strategy::SamplerDensityStrategy) = strategy.domain

StrategyInterface.sample_action(strategy::SamplerStrategy,
                                rng::AbstractRNG = Random.default_rng()) =
    strategy.sampler(rng)

StrategyInterface.sample_action(strategy::SamplerDensityStrategy,
                                rng::AbstractRNG = Random.default_rng()) =
    strategy.sampler(rng)

StrategyInterface.action_density(strategy::SamplerDensityStrategy, action) =
    strategy.density_fn(action)

end