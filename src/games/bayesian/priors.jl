module BayesianPriors

using Random

using ..Spaces
using ..StrategyInterface
using ..LocalStrategies

export AbstractPrior
export CommonPrior
export IndependentPrior

export type_space
export marginal_type_space
export sample_type_profile
export prior_probability
export marginal_probability

abstract type AbstractPrior end

# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

function _normalize_probs(probs)
    length(probs) > 0 || throw(ArgumentError("Probability support must be nonempty."))
    any(p -> p < 0, probs) && throw(ArgumentError("Probabilities must be nonnegative."))
    z = sum(probs)
    z > 0 || throw(ArgumentError("Probabilities must sum to a positive value."))
    return Float64.(probs) ./ z
end

function _canonicalize_type_profiles_probs(support_profiles, probs)
    length(support_profiles) == length(probs) ||
        throw(ArgumentError("Support and probabilities must have the same length."))
    isempty(support_profiles) &&
        throw(ArgumentError("Support must be nonempty."))

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

@inline function _sample_from_support_probs(support, probs, rng::AbstractRNG)
    r = rand(rng)
    c = 0.0
    @inbounds for i in eachindex(probs)
        c += probs[i]
        if r <= c
            return support[i]
        end
    end
    return support[last(eachindex(probs))]
end

# ----------------------------------------------------------------------
# Common prior
# ----------------------------------------------------------------------

struct CommonPrior{S,P} <: AbstractPrior
    support_profiles::S
    probs::P

    function CommonPrior(support_profiles, probs)
        profs, p = _canonicalize_type_profiles_probs(support_profiles, probs)
        return new{typeof(profs),typeof(p)}(profs, p)
    end
end

function sample_type_profile(prior::CommonPrior, rng::AbstractRNG = Random.default_rng())
    return _sample_from_support_probs(prior.support_profiles, prior.probs, rng)
end

function prior_probability(prior::CommonPrior, profile)
    S = prior.support_profiles
    P = prior.probs
    @inbounds for i in eachindex(P)
        if S[i] == profile
            return Float64(P[i])
        end
    end
    return 0.0
end

function marginal_probability(prior::CommonPrior, player::Int, typ)
    S = prior.support_profiles
    P = prior.probs
    length(S) > 0 || throw(ArgumentError("Prior support must be nonempty."))
    1 <= player <= length(first(S)) || throw(ArgumentError("Invalid player index $player."))

    acc = 0.0
    @inbounds for i in eachindex(P)
        if S[i][player] == typ
            acc += P[i]
        end
    end
    return acc
end

function marginal_type_space(prior::CommonPrior, player::Int)
    S = prior.support_profiles
    length(S) > 0 || throw(ArgumentError("Prior support must be nonempty."))
    1 <= player <= length(first(S)) || throw(ArgumentError("Invalid player index $player."))

    U = typeof(first(S)[player])
    @inbounds for i in 2:length(S)
        U = typejoin(U, typeof(S[i][player]))
    end

    vals = Vector{U}()
    seen = Set{U}()

    @inbounds for i in eachindex(S)
        t = convert(U, S[i][player])
        if !(t in seen)
            push!(seen, t)
            push!(vals, t)
        end
    end

    return Spaces.FiniteSpace(Tuple(vals))
end

type_space(prior::CommonPrior, player::Int) = marginal_type_space(prior, player)

# ----------------------------------------------------------------------
# Independent prior
# ----------------------------------------------------------------------

struct IndependentPrior{S,M} <: AbstractPrior
    spaces::S
    marginals::M

    function IndependentPrior(spaces::Tuple, marginals::Tuple)
        length(spaces) == length(marginals) ||
            throw(ArgumentError("Spaces and marginals must have the same length."))
        length(spaces) > 0 ||
            throw(ArgumentError("IndependentPrior requires at least one player."))

        @inbounds for i in eachindex(marginals)
            marginals[i] isa StrategyInterface.AbstractLocalStrategy ||
                throw(ArgumentError(
                    "Each marginal must be an AbstractLocalStrategy. Entry $i has type $(typeof(marginals[i]))."
                ))
        end

        return new{typeof(spaces),typeof(marginals)}(spaces, marginals)
    end
end

type_space(prior::IndependentPrior, player::Int) = prior.spaces[player]
marginal_type_space(prior::IndependentPrior, player::Int) = prior.spaces[player]

function sample_type_profile(prior::IndependentPrior, rng::AbstractRNG = Random.default_rng())
    N = length(prior.marginals)
    return ntuple(i -> StrategyInterface.sample_action(prior.marginals[i], rng), N)
end

function prior_probability(prior::IndependentPrior, profile::Tuple)
    length(profile) == length(prior.marginals) ||
        throw(ArgumentError("Type profile length does not match the prior."))

    p = 1.0
    @inbounds for i in eachindex(prior.marginals)
        p *= StrategyInterface.action_probability(prior.marginals[i], profile[i])
    end
    return p
end

function marginal_probability(prior::IndependentPrior, player::Int, typ)
    1 <= player <= length(prior.marginals) || throw(ArgumentError("Invalid player index $player."))
    return StrategyInterface.action_probability(prior.marginals[player], typ)
end

end