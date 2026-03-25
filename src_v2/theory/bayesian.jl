module Bayesian

using Random
using ..Spaces
using ..Strategies

export CommonPrior, IndependentPrior
export player_type, type_profile
export type_space, marginal_type_space
export sample_type_profile, prior_probability, marginal_probability

player_type(state, player::Int) =
    error("player_type is not implemented for $(typeof(state)).")

type_profile(state) =
    error("type_profile is not implemented for $(typeof(state)).")

struct CommonPrior{S,P}
    support_profiles::S
    probs::P
end

function CommonPrior(support_profiles, probs)
    length(support_profiles) == length(probs) ||
        throw(ArgumentError("Support and probabilities must have the same length."))
    p = Strategies.probabilities(Strategies.FiniteMixedStrategy(probs))
    return CommonPrior{typeof(support_profiles), typeof(p)}(support_profiles, p)
end

function sample_type_profile(prior::CommonPrior, rng::AbstractRNG = Random.default_rng())
    r = rand(rng)
    c = 0.0
    S = prior.support_profiles
    P = prior.probs
    @inbounds for i in eachindex(P)
        c += P[i]
        if r <= c
            return S[i]
        end
    end
    return S[last(eachindex(P))]
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
    T = typeof(S[first(eachindex(S))][player])
    vals = Vector{T}()
    seen = Set{T}()
    @inbounds for i in eachindex(S)
        t = S[i][player]
        if !(t in seen)
            push!(seen, t)
            push!(vals, t)
        end
    end
    return Spaces.FiniteSpace(vals)
end

type_space(prior::CommonPrior, player::Int) = marginal_type_space(prior, player)

struct IndependentPrior{S<:Tuple,M<:Tuple}
    spaces::S
    marginals::M
end

function IndependentPrior(spaces::Tuple, marginals::Tuple)
    length(spaces) == length(marginals) ||
        throw(ArgumentError("Spaces and marginals must have the same length."))
    return IndependentPrior{typeof(spaces), typeof(marginals)}(spaces, marginals)
end

type_space(prior::IndependentPrior, player::Int) = prior.spaces[player]
marginal_type_space(prior::IndependentPrior, player::Int) = prior.spaces[player]

function sample_type_profile(prior::IndependentPrior, rng::AbstractRNG = Random.default_rng())
    N = length(prior.marginals)
    return ntuple(i -> Strategies.sample_action(prior.marginals[i], rng), N)
end

function prior_probability(prior::IndependentPrior, profile::Tuple)
    length(profile) == length(prior.marginals) ||
        throw(ArgumentError("Type profile length does not match the prior."))
    p = 1.0
    @inbounds for i in eachindex(prior.marginals)
        p *= Strategies.probability(prior.marginals[i], profile[i])
    end
    return p
end

marginal_probability(prior::IndependentPrior, player::Int, typ) =
    Strategies.probability(prior.marginals[player], typ)

end