module Spaces

using Random

export AbstractSpace
export FiniteSpace, IndexedDiscreteSpace, BoxSpace, SimplexSpace, ProductSpace
export contains, sample, dimension

abstract type AbstractSpace end

"""
Explicit finite domain.
"""
struct FiniteSpace{T,C} <: AbstractSpace
    elements::C
    function FiniteSpace{T,C}(elements::C) where {T,C}
        isempty(elements) && throw(ArgumentError("FiniteSpace cannot be empty."))
        new{T,C}(elements)
    end
end

FiniteSpace(elements::C) where {T,C<:AbstractVector{T}} = FiniteSpace{T,C}(elements)

function FiniteSpace(elements::Tup) where {Tup<:Tuple}
    isempty(elements) && throw(ArgumentError("FiniteSpace cannot be empty."))

    T = typeof(elements[1])
    @inbounds for i in 2:length(elements)
        T = typejoin(T, typeof(elements[i]))
    end

    return FiniteSpace{T,Tup}(elements)
end

"""
Finite indexed domain `1:n`.
"""
struct IndexedDiscreteSpace <: AbstractSpace
    n::Int
    function IndexedDiscreteSpace(n::Int)
        n > 0 || throw(ArgumentError("IndexedDiscreteSpace size must be positive, got $n."))
        new(n)
    end
end

"""
Axis-aligned box in `R^d`.
"""
struct BoxSpace{T<:Real,V<:AbstractVector{T}} <: AbstractSpace
    low::V
    high::V
    function BoxSpace(low::V, high::V) where {T<:Real,V<:AbstractVector{T}}
        length(low) == length(high) ||
            throw(ArgumentError("BoxSpace low/high must have the same length."))
        @inbounds for i in eachindex(low, high)
            low[i] <= high[i] || throw(ArgumentError("BoxSpace requires low <= high elementwise."))
        end
        new{T,V}(low, high)
    end
end

"""
Probability simplex of dimension `n`.
"""
struct SimplexSpace <: AbstractSpace
    n::Int
    function SimplexSpace(n::Int)
        n > 0 || throw(ArgumentError("SimplexSpace dimension must be positive, got $n."))
        new(n)
    end
end

"""
Cartesian product of spaces.
"""
struct ProductSpace{S<:Tuple} <: AbstractSpace
    spaces::S
    function ProductSpace(spaces::S) where {S<:Tuple}
        length(spaces) > 0 || throw(ArgumentError("ProductSpace cannot be empty."))
        new{S}(spaces)
    end
end

dimension(::FiniteSpace) = nothing
dimension(s::IndexedDiscreteSpace) = s.n
dimension(s::BoxSpace) = length(s.low)
dimension(s::SimplexSpace) = s.n

function dimension(s::ProductSpace)
    dims = ntuple(i -> dimension(s.spaces[i]), length(s.spaces))
    all(d -> !isnothing(d), dims) || return nothing
    return sum(dims)
end

contains(space::FiniteSpace, x) = x in space.elements
contains(space::IndexedDiscreteSpace, x) = x isa Integer && 1 <= x <= space.n

function contains(space::BoxSpace, x)
    (x isa AbstractVector || x isa Tuple) || return false
    length(x) == length(space.low) || return false
    @inbounds for i in eachindex(space.low)
        xi = x[i]
        (space.low[i] <= xi <= space.high[i]) || return false
    end
    return true
end

function contains(space::SimplexSpace, x)
    (x isa AbstractVector || x isa Tuple) || return false
    length(x) == space.n || return false
    s = 0.0
    @inbounds for i in eachindex(x)
        xi = x[i]
        xi >= 0 || return false
        s += xi
    end
    return isapprox(s, 1.0; atol = 1e-8)
end

function contains(space::ProductSpace, x)
    applicable(length, x) || return false
    length(x) == length(space.spaces) || return false
    @inbounds for i in eachindex(space.spaces)
        applicable(getindex, x, i) || return false
        contains(space.spaces[i], x[i]) || return false
    end
    return true
end

sample(rng::AbstractRNG, s::FiniteSpace) = rand(rng, s.elements)
sample(rng::AbstractRNG, s::IndexedDiscreteSpace) = rand(rng, Base.OneTo(s.n))
sample(rng::AbstractRNG, s::BoxSpace) = s.low .+ rand(rng, length(s.low)) .* (s.high .- s.low)

function sample(rng::AbstractRNG, s::SimplexSpace)
    x = rand(rng, s.n)
    return x ./ sum(x)
end

sample(rng::AbstractRNG, s::ProductSpace) =
    ntuple(i -> sample(rng, s.spaces[i]), length(s.spaces))

end