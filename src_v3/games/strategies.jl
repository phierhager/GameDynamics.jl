module Strategies

using Random
using ..Kernel

export AbstractStrategy
export DeterministicStrategy
export FiniteMixedStrategy
export CorrelatedStrategy
export CallableBehaviorStrategy, TableBehaviorStrategy, DenseBehaviorStrategy, DenseVectorBehaviorStrategy
export SamplerStrategy, SamplerDensityStrategy
export StrategyProfile

export support, probabilities
export sample_action, sample_joint_action
export probability, density
export local_strategy
export expected_value, monte_carlo_expectation
export joint_probability, joint_density

abstract type AbstractStrategy end

# ----------------------------------------------------------------------
# Typed support canonicalization
# ----------------------------------------------------------------------

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
# Profiles
# ----------------------------------------------------------------------

"""
Tuple-backed fixed-player strategy profile.

The field type is left as a concrete tuple type to preserve heterogeneous
strategy types for inference. Constructor validation enforces that all entries
are strategies.
"""
struct StrategyProfile{N,S<:Tuple}
    strategies::S
end

function StrategyProfile(strategies::S) where {S<:Tuple}
    N = length(strategies)
    @inbounds for i in 1:N
        strategies[i] isa AbstractStrategy ||
            throw(ArgumentError("All entries of a StrategyProfile must subtype AbstractStrategy. Entry $i has type $(typeof(strategies[i]))."))
    end
    return StrategyProfile{N,S}(strategies)
end

Base.getindex(p::StrategyProfile, i::Int) = p.strategies[i]
Base.length(::StrategyProfile{N}) where {N} = N
Base.iterate(p::StrategyProfile, st...) = iterate(p.strategies, st...)
Base.eltype(::Type{<:StrategyProfile}) = AbstractStrategy
Base.firstindex(::StrategyProfile) = 1
Base.lastindex(p::StrategyProfile) = length(p)
Base.Tuple(p::StrategyProfile) = p.strategies

num_strategies(p::StrategyProfile{N}) where {N} = N
num_strategies(p::Tuple) = length(p)

# ----------------------------------------------------------------------
# Pure / deterministic strategies
# ----------------------------------------------------------------------

struct DeterministicStrategy{A} <: AbstractStrategy
    action::A
end

support(s::DeterministicStrategy) = (s.action,)
probabilities(::DeterministicStrategy) = (1.0,)

sample_action(s::DeterministicStrategy, rng::AbstractRNG = Random.default_rng()) = s.action
probability(s::DeterministicStrategy, action) = action == s.action ? 1.0 : 0.0

# ----------------------------------------------------------------------
# Finite mixed strategies
# ----------------------------------------------------------------------

struct FiniteMixedStrategy{A,P} <: AbstractStrategy
    actions::A
    probs::P
end

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

function FiniteMixedStrategy(actions, probs)
    acts, p = _canonicalize_support_probs(actions, probs)
    return FiniteMixedStrategy{typeof(acts), typeof(p)}(acts, p)
end

FiniteMixedStrategy(probs) = FiniteMixedStrategy(Base.OneTo(length(probs)), probs)

"""
Tuple-friendly constructor for small exact games.
"""
FiniteMixedStrategy(actions::Tuple, probs::Tuple) = FiniteMixedStrategy(actions, probs)

support(s::FiniteMixedStrategy) = s.actions
probabilities(s::FiniteMixedStrategy) = s.probs

function sample_action(s::FiniteMixedStrategy, rng::AbstractRNG = Random.default_rng())
    r = rand(rng)
    c = 0.0
    A = support(s)
    P = probabilities(s)
    @inbounds for i in eachindex(P)
        c += P[i]
        if r <= c
            return A[i]
        end
    end
    return A[last(eachindex(P))]
end

function probability(s::FiniteMixedStrategy, action)
    A = support(s)
    P = probabilities(s)
    @inbounds for i in eachindex(P)
        if A[i] == action
            return Float64(P[i])
        end
    end
    return 0.0
end

function expected_value(s::FiniteMixedStrategy, values)
    length(values) == length(probabilities(s)) ||
        throw(ArgumentError("Values must align with the strategy support."))
    acc = 0.0
    P = probabilities(s)
    @inbounds for i in eachindex(P)
        acc += P[i] * values[i]
    end
    return acc
end

# ----------------------------------------------------------------------
# Correlated strategies over finite joint support
# ----------------------------------------------------------------------

struct CorrelatedStrategy{S,P} <: AbstractStrategy
    support_profiles::S
    probs::P
end

function CorrelatedStrategy(support_profiles, probs)
    profs, p = _canonicalize_profiles_probs(support_profiles, probs)
    return CorrelatedStrategy{typeof(profs), typeof(p)}(profs, p)
end

support(s::CorrelatedStrategy) = s.support_profiles
probabilities(s::CorrelatedStrategy) = s.probs

function sample_action(s::CorrelatedStrategy, rng::AbstractRNG = Random.default_rng())
    r = rand(rng)
    c = 0.0
    S = support(s)
    P = probabilities(s)
    @inbounds for i in eachindex(P)
        c += P[i]
        if r <= c
            return S[i]
        end
    end
    return S[last(eachindex(P))]
end

function probability(s::CorrelatedStrategy, profile)
    S = support(s)
    P = probabilities(s)
    @inbounds for i in eachindex(P)
        if S[i] == profile
            return Float64(P[i])
        end
    end
    return 0.0
end

joint_probability(s::CorrelatedStrategy, profile) = probability(s, profile)

# ----------------------------------------------------------------------
# Behavior strategies
# ----------------------------------------------------------------------

struct CallableBehaviorStrategy{F} <: AbstractStrategy
    f::F
end

"""
Flexible dictionary-backed behavior table.
Good for convenience, not the fastest option.
"""
struct TableBehaviorStrategy{K,S,M<:AbstractDict{K,S}} <: AbstractStrategy
    table::M
end

"""
Dense integer-indexed behavior table for performance-sensitive code.
Assumes infosets are encoded as 1-based integers.

Tuple-backed variant: best for small fixed tables where preserving
full static structure is beneficial.
"""
struct DenseBehaviorStrategy{S,T<:Tuple{Vararg{S}}} <: AbstractStrategy
    table::T
end

"""
Vector-backed dense behavior table for larger encoded infoset maps.
Assumes infosets are encoded as 1-based integers.
"""
struct DenseVectorBehaviorStrategy{S,V<:AbstractVector{S}} <: AbstractStrategy
    table::V
end

local_strategy(s::CallableBehaviorStrategy, infoset) = s.f(infoset)

function local_strategy(s::TableBehaviorStrategy, infoset)
    haskey(s.table, infoset) || throw(KeyError("No local strategy stored for infoset $infoset."))
    return s.table[infoset]
end

local_strategy(s::DenseBehaviorStrategy, infoset::Int) = s.table[infoset]
local_strategy(s::DenseVectorBehaviorStrategy, infoset::Int) = s.table[infoset]

sample_action(s::CallableBehaviorStrategy, infoset, rng::AbstractRNG = Random.default_rng()) =
    sample_action(local_strategy(s, infoset), rng)

sample_action(s::TableBehaviorStrategy, infoset, rng::AbstractRNG = Random.default_rng()) =
    sample_action(local_strategy(s, infoset), rng)

sample_action(s::DenseBehaviorStrategy, infoset::Int, rng::AbstractRNG = Random.default_rng()) =
    sample_action(local_strategy(s, infoset), rng)

sample_action(s::DenseVectorBehaviorStrategy, infoset::Int, rng::AbstractRNG = Random.default_rng()) =
    sample_action(local_strategy(s, infoset), rng)

probability(s::CallableBehaviorStrategy, infoset, action) =
    probability(local_strategy(s, infoset), action)

probability(s::TableBehaviorStrategy, infoset, action) =
    probability(local_strategy(s, infoset), action)

probability(s::DenseBehaviorStrategy, infoset::Int, action) =
    probability(local_strategy(s, infoset), action)

probability(s::DenseVectorBehaviorStrategy, infoset::Int, action) =
    probability(local_strategy(s, infoset), action)

# ----------------------------------------------------------------------
# Continuous mixed strategies
# ----------------------------------------------------------------------

"""
Sampler-only continuous strategy.
Useful for rollout/sampling workflows.
"""
struct SamplerStrategy{S,Dom} <: AbstractStrategy
    sampler::S
    domain::Dom
end

support(s::SamplerStrategy) = s.domain
sample_action(s::SamplerStrategy, rng::AbstractRNG = Random.default_rng()) = s.sampler(rng)

"""
Sampler + density continuous strategy.
Useful when both simulation and density evaluation are needed.
"""
struct SamplerDensityStrategy{S,D,Dom} <: AbstractStrategy
    sampler::S
    density_fn::D
    domain::Dom
end

support(s::SamplerDensityStrategy) = s.domain
sample_action(s::SamplerDensityStrategy, rng::AbstractRNG = Random.default_rng()) = s.sampler(rng)
density(s::SamplerDensityStrategy, action) = s.density_fn(action)

density(strategy::AbstractStrategy, action) =
    throw(MethodError(density, (strategy, action)))

function monte_carlo_expectation(f, s::AbstractStrategy; rng::AbstractRNG = Random.default_rng(), n_samples::Int = 1024)
    n_samples > 0 || throw(ArgumentError("n_samples must be positive."))
    acc = 0.0
    for _ in 1:n_samples
        acc += f(sample_action(s, rng))
    end
    return acc / n_samples
end

# ----------------------------------------------------------------------
# Joint / profile utilities
# ----------------------------------------------------------------------

@inline function _sample_joint_action_tuple(profile::Tuple, rng::AbstractRNG)
    acts = ntuple(i -> sample_action(profile[i], rng), length(profile))
    return Kernel.joint_action(acts)
end

sample_joint_action(profile::StrategyProfile{N}, rng::AbstractRNG = Random.default_rng()) where {N} =
    _sample_joint_action_tuple(profile.strategies, rng)

sample_joint_action(profile::Tuple{Vararg{AbstractStrategy,N}}, rng::AbstractRNG = Random.default_rng()) where {N} =
    _sample_joint_action_tuple(profile, rng)

function sample_joint_action(profile::Tuple, rng::AbstractRNG = Random.default_rng())
    return sample_joint_action(StrategyProfile(profile), rng)
end

@inline function _joint_probability_tuple(profile::Tuple, actions::Tuple)
    length(profile) == length(actions) ||
        throw(ArgumentError("Profile and action tuple must have the same length."))
    p = 1.0
    @inbounds for i in eachindex(profile)
        p *= probability(profile[i], actions[i])
    end
    return p
end

joint_probability(profile::StrategyProfile, actions::Tuple) =
    _joint_probability_tuple(profile.strategies, actions)

joint_probability(profile::Tuple{Vararg{AbstractStrategy,N}}, actions::Tuple) where {N} =
    _joint_probability_tuple(profile, actions)

function joint_probability(profile::Tuple, actions::Tuple)
    return joint_probability(StrategyProfile(profile), actions)
end

@inline function _joint_density_tuple(profile::Tuple, actions::Tuple)
    length(profile) == length(actions) ||
        throw(ArgumentError("Profile and action tuple must have the same length."))
    d = 1.0
    @inbounds for i in eachindex(profile)
        d *= density(profile[i], actions[i])
    end
    return d
end

joint_density(profile::StrategyProfile, actions::Tuple) =
    _joint_density_tuple(profile.strategies, actions)

joint_density(profile::Tuple{Vararg{AbstractStrategy,N}}, actions::Tuple) where {N} =
    _joint_density_tuple(profile, actions)

function joint_density(profile::Tuple, actions::Tuple)
    return joint_density(StrategyProfile(profile), actions)
end

end