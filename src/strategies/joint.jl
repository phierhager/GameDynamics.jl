module JointStrategies

using Random
using ..StrategyInterface
using ..StrategyInternalUtils

export CorrelatedRecommendationDevice

struct CorrelatedRecommendationDevice{S,P} <: StrategyInterface.AbstractLocalStrategy
    joint_tuples::S
    probs::P

    function CorrelatedRecommendationDevice(joint_tuples, probs)
        tuples, p = StrategyInternalUtils.canonicalize_joint_tuple_probs(joint_tuples, probs)
        new{typeof(tuples), typeof(p)}(tuples, p)
    end
end

StrategyInterface.support(strategy::CorrelatedRecommendationDevice) = strategy.joint_tuples
StrategyInterface.probabilities(strategy::CorrelatedRecommendationDevice) = strategy.probs

function StrategyInterface.sample_action(strategy::CorrelatedRecommendationDevice,
                                         rng::AbstractRNG = Random.default_rng())
    r = rand(rng)
    c = 0.0
    S = StrategyInterface.support(strategy)
    P = StrategyInterface.probabilities(strategy)
    @inbounds for i in eachindex(P)
        c += P[i]
        if r <= c
            return S[i]
        end
    end
    return S[last(eachindex(P))]
end

function StrategyInterface.action_probability(strategy::CorrelatedRecommendationDevice, joint_tuple)
    S = StrategyInterface.support(strategy)
    P = StrategyInterface.probabilities(strategy)
    @inbounds for i in eachindex(P)
        if S[i] == joint_tuple
            return Float64(P[i])
        end
    end
    return 0.0
end

end