module Bayesian

using Random
using ..Spaces
using ..Strategies

export CommonPrior, IndependentPrior
export player_type, type_profile
export type_space, marginal_type_space
export sample_type_profile, prior_probability, marginal_probability

"""
Return the type of `player` from a game state.

Concrete state representations for Bayesian games are expected to specialize this.
"""
player_type(state, player::Int) =
    error("player_type is not implemented for $(typeof(state)).")

"""
Return the full type profile from a game state.

Concrete state representations for Bayesian games are expected to specialize this.
"""
type_profile(state) =
    error("type_profile is not implemented for $(typeof(state)).")

# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

"""
Canonicalize a finite-support distribution over type profiles.

Behavior:
- validates support/probability length agreement
- rejects empty support
- normalizes probabilities to sum to one
- merges duplicate profiles by summing their probability mass
- preserves first-seen support order among distinct profiles
"""
function _canonicalize_type_profiles_probs(support_profiles, probs)
    length(support_profiles) == length(probs) ||
        throw(ArgumentError("Support and probabilities must have the same length."))
    isempty(support_profiles) &&
        throw(ArgumentError("CommonPrior support must be nonempty."))

    p = Strategies.probabilities(Strategies.FiniteMixedStrategy(probs))

    T = eltype(support_profiles)
    acc = Dict{T,Float64}()
    order = Vector{T}()
    sizehint!(order, min(length(support_profiles), 8))

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

@inline function _validate_profile_player_index(first_profile, player::Int, who::AbstractString)
    1 <= player <= length(first_profile) ||
        throw(ArgumentError(
            "Invalid player index $player for $who of length $(length(first_profile))."
        ))
    return player
end

# ----------------------------------------------------------------------
# Common prior over finite type profiles
# ----------------------------------------------------------------------

"""
Finite common prior over type profiles.

This is a semantic object representing a probability distribution over full player
type profiles.

Construction semantics:
- probabilities are normalized automatically
- duplicate support profiles are merged by summing their probability mass
- support order for distinct profiles follows first appearance in the input
"""
struct CommonPrior{S,P}
    support_profiles::S
    probs::P
end

function CommonPrior(support_profiles, probs)
    profs, p = _canonicalize_type_profiles_probs(support_profiles, probs)
    return CommonPrior{typeof(profs), typeof(p)}(profs, p)
end

"""
Sample a type profile from the common prior.
"""
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

"""
Return the probability mass assigned to a full type profile.

Returns `0.0` when the profile is not in the support.
"""
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

"""
Return the marginal probability that `player` has type `typ`.
"""
function marginal_probability(prior::CommonPrior, player::Int, typ)
    S = prior.support_profiles
    P = prior.probs

    length(S) > 0 || throw(ArgumentError("Prior support must be nonempty."))

    first_profile = first(S)
    _validate_profile_player_index(first_profile, player, "type profiles")

    acc = 0.0
    @inbounds for i in eachindex(P)
        if S[i][player] == typ
            acc += P[i]
        end
    end
    return acc
end

"""
Return the finite marginal type space for `player` induced by the common prior.

The returned support preserves first-seen order from the prior support profiles.
"""
function marginal_type_space(prior::CommonPrior, player::Int)
    S = prior.support_profiles
    length(S) > 0 || throw(ArgumentError("Prior support must be nonempty."))

    first_profile = first(S)
    _validate_profile_player_index(first_profile, player, "type profiles")

    U = typeof(first_profile[player])

    vals = Vector{U}()
    seen = Set{U}()
    sizehint!(vals, min(length(S), 8))

    @inbounds for i in eachindex(S)
        t = S[i][player]
        if !(t in seen)
            push!(seen, t)
            push!(vals, t)
        end
    end

    return Spaces.FiniteSpace(Tuple(vals))
end

"""
Return the type space for `player` induced by the prior.

For `CommonPrior`, this is the same as `marginal_type_space`.
"""
type_space(prior::CommonPrior, player::Int) = marginal_type_space(prior, player)

# ----------------------------------------------------------------------
# Independent prior
# ----------------------------------------------------------------------

"""
Product prior with independent player marginals.

Fields:
- `spaces`: per-player type spaces
- `marginals`: per-player type distributions/strategies

The intended semantics are independent type draws across players.
"""
struct IndependentPrior{S<:Tuple,M<:Tuple}
    spaces::S
    marginals::M
end

function IndependentPrior(spaces::Tuple, marginals::Tuple)
    length(spaces) == length(marginals) ||
        throw(ArgumentError("Spaces and marginals must have the same length."))
    length(spaces) > 0 ||
        throw(ArgumentError("IndependentPrior requires at least one player."))

    @inbounds for i in eachindex(marginals)
        marginals[i] isa Strategies.AbstractStrategy ||
            throw(ArgumentError(
                "Each marginal must subtype AbstractStrategy. Entry $i has type $(typeof(marginals[i]))."
            ))
    end

    return IndependentPrior{typeof(spaces), typeof(marginals)}(spaces, marginals)
end

"""
Return the declared type space for `player`.
"""
type_space(prior::IndependentPrior, player::Int) = prior.spaces[player]

"""
Return the declared marginal type space for `player`.
"""
marginal_type_space(prior::IndependentPrior, player::Int) = prior.spaces[player]

"""
Sample a full type profile under independent marginals.
"""
function sample_type_profile(prior::IndependentPrior, rng::AbstractRNG = Random.default_rng())
    N = length(prior.marginals)
    return ntuple(i -> Strategies.sample_action(prior.marginals[i], rng), N)
end

"""
Return the probability of a full type profile under the independent prior.
"""
function prior_probability(prior::IndependentPrior, profile::Tuple)
    length(profile) == length(prior.marginals) ||
        throw(ArgumentError("Type profile length does not match the prior."))

    p = 1.0
    @inbounds for i in eachindex(prior.marginals)
        p *= Strategies.probability(prior.marginals[i], profile[i])
    end
    return p
end

"""
Return the marginal probability that `player` has type `typ`.
"""
function marginal_probability(prior::IndependentPrior, player::Int, typ)
    1 <= player <= length(prior.marginals) ||
        throw(ArgumentError(
            "Invalid player index $player for IndependentPrior with $(length(prior.marginals)) marginals."
        ))
    return Strategies.probability(prior.marginals[player], typ)
end

end