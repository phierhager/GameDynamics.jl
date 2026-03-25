module Spaces

using Random

export AbstractSpace, FiniteSpace, IndexedDiscreteSpace, BoxSpace, SimplexSpace, ProductSpace
export contains, sample, dimension

abstract type AbstractSpace end

"""
Explicit finite domain. Good for metadata and small spaces.

The container may be a tuple or an abstract vector.
"""
struct FiniteSpace{T,C} <: AbstractSpace
    elements::C
end

FiniteSpace(elements::C) where {C<:Tuple} = FiniteSpace{eltype(elements), C}(elements)
FiniteSpace(elements::C) where {T,C<:AbstractVector{T}} = FiniteSpace{T, C}(elements)

"""
Indexed discrete domain of size n.
Useful for indexed-action games and masks.
"""
struct IndexedDiscreteSpace <: AbstractSpace
    n::Int
end

struct BoxSpace{T<:Real,V<:AbstractVector{T}} <: AbstractSpace
    low::V
    high::V
end

struct SimplexSpace <: AbstractSpace
    n::Int
end

struct ProductSpace{S<:Tuple} <: AbstractSpace
    spaces::S
end

dimension(::FiniteSpace) = nothing
dimension(s::IndexedDiscreteSpace) = s.n
dimension(s::BoxSpace) = length(s.low)
dimension(s::SimplexSpace) = s.n
function dimension(s::ProductSpace)
    dims = ntuple(i -> dimension(s.spaces[i]), length(s.spaces))
    all(!isnothing, dims) || return nothing
    return sum(dims)
end

contains(space::FiniteSpace, x) = x in space.elements
contains(space::IndexedDiscreteSpace, x) = x isa Integer && 1 <= x <= space.n
contains(space::BoxSpace, x) =
    length(x) == length(space.low) &&
    all(space.low .<= x) &&
    all(x .<= space.high)
contains(space::SimplexSpace, x) =
    length(x) == space.n && all(x .>= 0) && isapprox(sum(x), 1.0; atol = 1e-8)
contains(space::ProductSpace, x) =
    length(x) == length(space.spaces) &&
    all(contains(space.spaces[i], x[i]) for i in eachindex(space.spaces))

sample(rng::AbstractRNG, s::FiniteSpace) = rand(rng, s.elements)
sample(rng::AbstractRNG, s::IndexedDiscreteSpace) = rand(rng, 1:s.n)
sample(rng::AbstractRNG, s::BoxSpace) = s.low .+ rand(rng, length(s.low)) .* (s.high .- s.low)

function sample(rng::AbstractRNG, s::SimplexSpace)
    x = rand(rng, s.n)
    return x ./ sum(x)
end

"""
Return a tuple to preserve product structure and type stability.
"""
sample(rng::AbstractRNG, s::ProductSpace) =
    ntuple(i -> sample(rng, s.spaces[i]), length(s.spaces))

end