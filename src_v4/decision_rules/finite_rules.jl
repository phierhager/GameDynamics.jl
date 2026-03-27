module FiniteDecisionRules

using Random
using ..DecisionRulesInterface

export DeterministicDecisionRule
export FiniteMixedDecisionRule

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

function _canonicalize_support_probs(actions::A, probs) where {A}
    length(actions) == length(probs) ||
        throw(ArgumentError("Actions and probabilities must have the same length."))
    isempty(actions) && throw(ArgumentError("Support must be nonempty."))

    p = _normalize_probs(probs)

    T = eltype(actions)
    acc = Dict{T,Float64}()
    order = Vector{T}()

    @inbounds for i in eachindex(actions)
        a = actions[i]
        if !haskey(acc, a)
            push!(order, a)
            acc[a] = 0.0
        end
        acc[a] += p[i]
    end

    acts = Tuple(order)
    ps = ntuple(i -> acc[order[i]], length(order))
    return acts, ps
end

# ----------------------------------------------------------------------
# Deterministic
# ----------------------------------------------------------------------

struct DeterministicDecisionRule{A} <: DecisionRulesInterface.AbstractDecisionRule
    action::A
end

DecisionRulesInterface.conditioning_kind(::Type{<:DeterministicDecisionRule}) =
    DecisionRulesInterface.NoConditioning()

DecisionRulesInterface.memory_class(::Type{<:DeterministicDecisionRule}) =
    DecisionRulesInterface.Memoryless()

DecisionRulesInterface.support(rule::DeterministicDecisionRule) = (rule.action,)
DecisionRulesInterface.probabilities(::DeterministicDecisionRule) = (1.0,)

DecisionRulesInterface.sample_action(rule::DeterministicDecisionRule,
                                     rng::AbstractRNG = Random.default_rng()) =
    rule.action

DecisionRulesInterface.action_probability(rule::DeterministicDecisionRule, action) =
    action == rule.action ? 1.0 : 0.0

# ----------------------------------------------------------------------
# Finite mixed
# ----------------------------------------------------------------------

struct FiniteMixedDecisionRule{A,P} <: DecisionRulesInterface.AbstractDecisionRule
    actions::A
    probs::P
end

function FiniteMixedDecisionRule(actions, probs)
    acts, p = _canonicalize_support_probs(actions, probs)
    return FiniteMixedDecisionRule{typeof(acts),typeof(p)}(acts, p)
end

FiniteMixedDecisionRule(probs) = FiniteMixedDecisionRule(Base.OneTo(length(probs)), probs)

FiniteMixedDecisionRule(actions::Tuple, probs::Tuple) =
    FiniteMixedDecisionRule(actions, probs)

DecisionRulesInterface.conditioning_kind(::Type{<:FiniteMixedDecisionRule}) =
    DecisionRulesInterface.NoConditioning()

DecisionRulesInterface.memory_class(::Type{<:FiniteMixedDecisionRule}) =
    DecisionRulesInterface.Memoryless()

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

end