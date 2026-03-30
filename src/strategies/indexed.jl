module IndexedStrategies

using Random
using ..StrategyInterface

export CallableIndexedStrategy
export TableIndexedStrategy
export DenseIndexedStrategy
export DenseVectorIndexedStrategy

"""
Callable indexed strategy.

`f(context)` must return a local strategy.
"""
struct CallableIndexedStrategy{
    CK<:StrategyInterface.AbstractContextKind,
    F,
    ISC<:StrategyInterface.AbstractInternalStateClass,
} <: StrategyInterface.AbstractStrategy
    f::F
    internal_state::ISC
end

"""
Dictionary-backed indexed strategy with explicit key and value types.
"""
struct TableIndexedStrategy{
    CK<:StrategyInterface.AbstractContextKind,
    K,
    S<:StrategyInterface.AbstractStrategy,
    T<:AbstractDict{K,S},
    ISC<:StrategyInterface.AbstractInternalStateClass,
} <: StrategyInterface.AbstractStrategy
    table::T
    internal_state::ISC
end

"""
Tuple-backed dense indexed strategy for small fixed integer-indexed maps.
Assumes contexts are encoded as 1-based integers.
"""
struct DenseIndexedStrategy{
    CK<:StrategyInterface.AbstractContextKind,
    T<:Tuple,
    ISC<:StrategyInterface.AbstractInternalStateClass,
} <: StrategyInterface.AbstractStrategy
    table::T
    internal_state::ISC
end
"""
Vector-backed dense indexed strategy for larger integer-indexed maps.
Assumes contexts are encoded as 1-based integers.
"""
struct DenseVectorIndexedStrategy{
    CK<:StrategyInterface.AbstractContextKind,
    S<:StrategyInterface.AbstractStrategy,
    V<:AbstractVector{S},
    ISC<:StrategyInterface.AbstractInternalStateClass,
} <: StrategyInterface.AbstractStrategy
    table::V
    internal_state::ISC
end

CallableIndexedStrategy(::Type{CK}, f;
                   internal_state::StrategyInterface.AbstractInternalStateClass = StrategyInterface.Stateless()) where {CK<:StrategyInterface.AbstractContextKind} =
    CallableIndexedStrategy{CK,typeof(f),typeof(internal_state)}(f, internal_state)

TableIndexedStrategy(::Type{CK}, table::T;
                internal_state::StrategyInterface.AbstractInternalStateClass = StrategyInterface.Stateless()) where {CK<:StrategyInterface.AbstractContextKind,K,S<:StrategyInterface.AbstractStrategy,T<:AbstractDict{K,S}} =
    TableIndexedStrategy{CK,K,S,T,typeof(internal_state)}(table, internal_state)

function DenseIndexedStrategy(::Type{CK}, table::T;
                         internal_state::StrategyInterface.AbstractInternalStateClass = StrategyInterface.Stateless()) where
    {CK<:StrategyInterface.AbstractContextKind,T<:Tuple}
    @inbounds for i in eachindex(table)
        table[i] isa StrategyInterface.AbstractStrategy ||
            throw(ArgumentError("DenseIndexedStrategy entries must be strategies. Entry $i has type $(typeof(table[i]))."))
    end
    return DenseIndexedStrategy{CK,T,typeof(internal_state)}(table, internal_state)
end

function DenseVectorIndexedStrategy(::Type{CK}, table::V;
                               internal_state::StrategyInterface.AbstractInternalStateClass = StrategyInterface.Stateless()) where {CK<:StrategyInterface.AbstractContextKind,S,V<:AbstractVector{S}}
    @inbounds for i in eachindex(table)
        table[i] isa StrategyInterface.AbstractStrategy ||
            throw(ArgumentError("DenseVectorIndexedStrategy entries must be strategies. Entry $i has type $(typeof(table[i]))."))
    end
    return DenseVectorIndexedStrategy{CK,S,V,typeof(internal_state)}(table, internal_state)
end

StrategyInterface.context_kind(::Type{<:CallableIndexedStrategy{CK}}) where {CK} = CK()
StrategyInterface.context_kind(::Type{<:TableIndexedStrategy{CK}}) where {CK} = CK()
StrategyInterface.context_kind(::Type{<:DenseIndexedStrategy{CK}}) where {CK} = CK()
StrategyInterface.context_kind(::Type{<:DenseVectorIndexedStrategy{CK}}) where {CK} = CK()

StrategyInterface.internal_state_class(::Type{<:CallableIndexedStrategy{CK,F,ISC}}) where {CK,F,ISC<:StrategyInterface.AbstractInternalStateClass} =
    ISC()
StrategyInterface.internal_state_class(::Type{<:TableIndexedStrategy{CK,K,S,T,ISC}}) where {CK,K,S,T,ISC<:StrategyInterface.AbstractInternalStateClass} =
    ISC()
StrategyInterface.internal_state_class(::Type{<:DenseIndexedStrategy{CK,T,ISC}}) where {CK,S,T,ISC<:StrategyInterface.AbstractInternalStateClass} =
    ISC()
StrategyInterface.internal_state_class(::Type{<:DenseVectorIndexedStrategy{CK,S,V,ISC}}) where {CK,S,V,ISC<:StrategyInterface.AbstractInternalStateClass} =
    ISC()

function StrategyInterface.local_strategy(strategy::CallableIndexedStrategy, context)
    substrategy = strategy.f(context)
    substrategy isa StrategyInterface.AbstractStrategy ||
        throw(ArgumentError("CallableIndexedStrategy must return an AbstractStrategy for context $context."))
    return substrategy
end

function StrategyInterface.local_strategy(strategy::TableIndexedStrategy, context)
    haskey(strategy.table, context) ||
        throw(KeyError("No local strategy stored for context $context."))
    return strategy.table[context]
end

StrategyInterface.local_strategy(strategy::DenseIndexedStrategy, context::Int) =
    strategy.table[context]

StrategyInterface.local_strategy(strategy::DenseVectorIndexedStrategy, context::Int) =
    strategy.table[context]

StrategyInterface.sample_action(strategy::CallableIndexedStrategy,
                                     context,
                                     rng::AbstractRNG = Random.default_rng()) =
    StrategyInterface.sample_action(
        StrategyInterface.local_strategy(strategy, context), rng
    )

StrategyInterface.sample_action(strategy::TableIndexedStrategy,
                                     context,
                                     rng::AbstractRNG = Random.default_rng()) =
    StrategyInterface.sample_action(
        StrategyInterface.local_strategy(strategy, context), rng
    )

StrategyInterface.sample_action(strategy::DenseIndexedStrategy,
                                     context::Int,
                                     rng::AbstractRNG = Random.default_rng()) =
    StrategyInterface.sample_action(
        StrategyInterface.local_strategy(strategy, context), rng
    )

StrategyInterface.sample_action(strategy::DenseVectorIndexedStrategy,
                                     context::Int,
                                     rng::AbstractRNG = Random.default_rng()) =
    StrategyInterface.sample_action(
        StrategyInterface.local_strategy(strategy, context), rng
    )

StrategyInterface.action_probability(strategy::CallableIndexedStrategy, context, action) =
    StrategyInterface.action_probability(
        StrategyInterface.local_strategy(strategy, context), action
    )

StrategyInterface.action_probability(strategy::TableIndexedStrategy, context, action) =
    StrategyInterface.action_probability(
        StrategyInterface.local_strategy(strategy, context), action
    )

StrategyInterface.action_probability(strategy::DenseIndexedStrategy, context::Int, action) =
    StrategyInterface.action_probability(
        StrategyInterface.local_strategy(strategy, context), action
    )

StrategyInterface.action_probability(strategy::DenseVectorIndexedStrategy, context::Int, action) =
    StrategyInterface.action_probability(
        StrategyInterface.local_strategy(strategy, context), action
    )

StrategyInterface.action_density(strategy::CallableIndexedStrategy, context, action) =
    StrategyInterface.action_density(
        StrategyInterface.local_strategy(strategy, context), action
    )

StrategyInterface.action_density(strategy::TableIndexedStrategy, context, action) =
    StrategyInterface.action_density(
        StrategyInterface.local_strategy(strategy, context), action
    )

StrategyInterface.action_density(strategy::DenseIndexedStrategy, context::Int, action) =
    StrategyInterface.action_density(
        StrategyInterface.local_strategy(strategy, context), action
    )

StrategyInterface.action_density(strategy::DenseVectorIndexedStrategy, context::Int, action) =
    StrategyInterface.action_density(
        StrategyInterface.local_strategy(strategy, context), action
    )

end