module IndexedStrategies

using Random
using ..StrategyInterface

export TableStrategy
export DenseTupleStrategy
export DenseVectorStrategy

"""
Dictionary-backed strategy:
- `extractor(record) -> key`
- `table[key] -> local strategy`
"""
struct TableStrategy{E,T} <:
       StrategyInterface.AbstractRecordStrategy
    extractor::E
    table::T
end

function TableStrategy(extractor, table)
    return TableStrategy{typeof(extractor),typeof(table)}(
        extractor, table
    )
end

function _local_strategy(strategy::TableStrategy, record)
    key = strategy.extractor(record)
    haskey(strategy.table, key) ||
        throw(KeyError("No local strategy stored for key $key."))
    substrategy = strategy.table[key]
    substrategy isa StrategyInterface.AbstractLocalStrategy ||
        throw(ArgumentError("Stored entries must be AbstractLocalStrategy values."))
    return substrategy
end

function StrategyInterface.sample_action(strategy::TableStrategy,
                                         record,
                                         rng::AbstractRNG = Random.default_rng())
    return StrategyInterface.sample_action(_local_strategy(strategy, record), rng)
end

function StrategyInterface.action_probability(strategy::TableStrategy, record, action)
    return StrategyInterface.action_probability(_local_strategy(strategy, record), action)
end

function StrategyInterface.action_density(strategy::TableStrategy, record, action)
    return StrategyInterface.action_density(_local_strategy(strategy, record), action)
end

"""
Tuple-backed dense strategy:
- `extractor(record) -> Int`
- `table[i] -> local strategy`
"""
struct DenseTupleStrategy{E,T<:Tuple} <:
       StrategyInterface.AbstractRecordStrategy
    extractor::E
    table::T
end

function DenseTupleStrategy(extractor, table::T) where {T<:Tuple}
    @inbounds for i in eachindex(table)
        table[i] isa StrategyInterface.AbstractLocalStrategy ||
            throw(ArgumentError("DenseTupleStrategy entries must be AbstractLocalStrategy. Entry $i has type $(typeof(table[i]))."))
    end
    return DenseTupleStrategy{typeof(extractor),T}(
        extractor, table
    )
end

function _local_strategy(strategy::DenseTupleStrategy, record)
    i = strategy.extractor(record)
    i isa Int || throw(ArgumentError("DenseTupleStrategy extractor must return Int."))
    return strategy.table[i]
end

function StrategyInterface.sample_action(strategy::DenseTupleStrategy,
                                         record,
                                         rng::AbstractRNG = Random.default_rng())
    return StrategyInterface.sample_action(_local_strategy(strategy, record), rng)
end

function StrategyInterface.action_probability(strategy::DenseTupleStrategy, record, action)
    return StrategyInterface.action_probability(_local_strategy(strategy, record), action)
end

function StrategyInterface.action_density(strategy::DenseTupleStrategy, record, action)
    return StrategyInterface.action_density(_local_strategy(strategy, record), action)
end

"""
Vector-backed dense strategy:
- `extractor(record) -> Int`
- `table[i] -> local strategy`
"""
struct DenseVectorStrategy{E,V} <:
       StrategyInterface.AbstractRecordStrategy
    extractor::E
    table::V
end

function DenseVectorStrategy(extractor, table::V) where {V<:AbstractVector}
    @inbounds for i in eachindex(table)
        table[i] isa StrategyInterface.AbstractLocalStrategy ||
            throw(ArgumentError("DenseVectorStrategy entries must be AbstractLocalStrategy. Entry $i has type $(typeof(table[i]))."))
    end
    return DenseVectorStrategy{typeof(extractor),V}(
        extractor, table
    )
end

function _local_strategy(strategy::DenseVectorStrategy, record)
    i = strategy.extractor(record)
    i isa Int || throw(ArgumentError("DenseVectorStrategy extractor must return Int."))
    return strategy.table[i]
end

function StrategyInterface.sample_action(strategy::DenseVectorStrategy,
                                         record,
                                         rng::AbstractRNG = Random.default_rng())
    return StrategyInterface.sample_action(_local_strategy(strategy, record), rng)
end

function StrategyInterface.action_probability(strategy::DenseVectorStrategy, record, action)
    return StrategyInterface.action_probability(_local_strategy(strategy, record), action)
end

function StrategyInterface.action_density(strategy::DenseVectorStrategy, record, action)
    return StrategyInterface.action_density(_local_strategy(strategy, record), action)
end

end