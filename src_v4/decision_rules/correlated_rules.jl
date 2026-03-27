module CorrelatedDecisionRules

using Random
using ..DecisionRulesInterface

export CorrelatedDecisionRule

# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

function _normalize_probs(probs::Tuple)
    length(probs) > 0 || throw(ArgumentError("Probability tuple must be nonempty."))
    any(p -> p < 0, probs) && throw(ArgumentError("Probabilities must be nonnegative."))
    z = sum(probs)
    z > 0 || throw(ArgumentError("Probabilities must sum to a positive value."))
    return ntuple(i -> Float64(probs[i]) / z, length(probs))
end

function _normalize_probs(probs::AbstractVector)
    isempty(probs) && throw(ArgumentError("Probability vector must be nonempty."))
    any(p -> p < 0, probs) && throw(ArgumentError("Probabilities must be nonnegative."))
    z = sum(probs)
    z > 0 || throw(ArgumentError("Probabilities must sum to a positive value."))
    return Float64.(probs) ./ z
end

function _canonicalize_profiles_probs(support_profiles::S, probs) where {S}
    length(support_profiles) == length(probs) ||
        throw(ArgumentError("Support and probabilities must have the same length."))
    isempty(support_profiles) && throw(ArgumentError("Support must be nonempty."))

    p = _normalize_probs(probs)

    T = eltype(support_profiles)
    acc = Dict{T,Float64}()
    order = Vector{T}()

    @inbounds for i in eachindex(support_profiles)
        prof = support_profiles[i]
        if !haskey(acc, prof)
            push!(order, prof)
            acc[prof] = 0.0
        end
        acc[prof] += p[i]
    end

    profs = Tuple(order)
    ps = ntuple(i -> acc[order[i]], length(order))
    return profs, ps
end

# ----------------------------------------------------------------------
# Correlated rule
# ----------------------------------------------------------------------

"""
Finite correlated decision rule over joint profiles.

This is a joint recommendation / joint-action distribution, not a local single-agent
rule. It still belongs in the decision-rule subsystem because it is a direct
behavioral object.
"""
struct CorrelatedDecisionRule{S,P} <: DecisionRulesInterface.AbstractDecisionRule
    support_profiles::S
    probs::P
end

function CorrelatedDecisionRule(support_profiles, probs)
    profs, p = _canonicalize_profiles_probs(support_profiles, probs)
    return CorrelatedDecisionRule{typeof(profs),typeof(p)}(profs, p)
end

DecisionRulesInterface.conditioning_kind(::Type{<:CorrelatedDecisionRule}) =
    DecisionRulesInterface.NoConditioning()

DecisionRulesInterface.memory_class(::Type{<:CorrelatedDecisionRule}) =
    DecisionRulesInterface.Memoryless()

DecisionRulesInterface.support(rule::CorrelatedDecisionRule) = rule.support_profiles
DecisionRulesInterface.probabilities(rule::CorrelatedDecisionRule) = rule.probs

function DecisionRulesInterface.sample_action(rule::CorrelatedDecisionRule,
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

function DecisionRulesInterface.action_probability(rule::CorrelatedDecisionRule, profile)
    S = DecisionRulesInterface.support(rule)
    P = DecisionRulesInterface.probabilities(rule)
    @inbounds for i in eachindex(P)
        if S[i] == profile
            return Float64(P[i])
        end
    end
    return 0.0
end

end