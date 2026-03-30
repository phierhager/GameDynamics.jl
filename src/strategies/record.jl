module RecordStrategies

using Random
using ..StrategyInterface

export DirectRecordStrategy
export ProjectedStrategy
export ExtractedCallableStrategy

# ----------------------------------------------------------------------
# Direct record strategy
# ----------------------------------------------------------------------

"""
A strategy that directly consumes the whole runtime record.

Expected signatures:
- `sampler(record, rng)`

Optional:
- `likelihood(record, action)`
- `density(record, action)`
"""
struct DirectRecordStrategy{S,L,D} <:
       StrategyInterface.AbstractRecordStrategy
    sampler::S
    likelihood::L
    density_fn::D
end

function DirectRecordStrategy(sampler;
                              likelihood=nothing,
                              density_fn=nothing)
    return DirectRecordStrategy{
        typeof(sampler), typeof(likelihood), typeof(density_fn)
    }(sampler, likelihood, density_fn)
end

function StrategyInterface.sample_action(strategy::DirectRecordStrategy,
                                         record,
                                         rng::AbstractRNG = Random.default_rng())
    return strategy.sampler(record, rng)
end

function StrategyInterface.action_probability(strategy::DirectRecordStrategy, record, action)
    strategy.likelihood === nothing &&
        throw(MethodError(StrategyInterface.action_probability, (strategy, record, action)))
    return strategy.likelihood(record, action)
end

function StrategyInterface.action_density(strategy::DirectRecordStrategy, record, action)
    strategy.density_fn === nothing &&
        throw(MethodError(StrategyInterface.action_density, (strategy, record, action)))
    return strategy.density_fn(record, action)
end

# ----------------------------------------------------------------------
# Extractor -> local strategy
# ----------------------------------------------------------------------

"""
A strategy that extracts a key/value from a runtime record and passes it to
a builder that returns a local strategy.

Expected:
- `extractor(record) -> x`
- `builder(x) -> local_strategy`
"""
struct ProjectedStrategy{E,B} <:
       StrategyInterface.AbstractRecordStrategy
    extractor::E
    builder::B
end

function ProjectedStrategy(extractor, builder)
    return ProjectedStrategy{typeof(extractor),typeof(builder)}(
        extractor, builder
    )
end

function _local_strategy(strategy::ProjectedStrategy, record)
    x = strategy.extractor(record)
    substrategy = strategy.builder(x)
    substrategy isa StrategyInterface.AbstractLocalStrategy ||
        throw(ArgumentError("ProjectedStrategy builder must return an AbstractLocalStrategy."))
    return substrategy
end

function StrategyInterface.sample_action(strategy::ProjectedStrategy,
                                         record,
                                         rng::AbstractRNG = Random.default_rng())
    substrategy = _local_strategy(strategy, record)
    return StrategyInterface.sample_action(substrategy, rng)
end

function StrategyInterface.action_probability(strategy::ProjectedStrategy, record, action)
    substrategy = _local_strategy(strategy, record)
    return StrategyInterface.action_probability(substrategy, action)
end

function StrategyInterface.action_density(strategy::ProjectedStrategy, record, action)
    substrategy = _local_strategy(strategy, record)
    return StrategyInterface.action_density(substrategy, action)
end

# ----------------------------------------------------------------------
# Extractor -> callable local law
# ----------------------------------------------------------------------

"""
Convenience wrapper:
- `extractor(record) -> x`
- `sampler(x, rng) -> action`

Optional:
- `likelihood(x, action)`
- `density(x, action)`

This is just a more direct version when you do not want to explicitly build a
local strategy object.
"""
struct ExtractedCallableStrategy{E,S,L,D} <:
       StrategyInterface.AbstractRecordStrategy
    extractor::E
    sampler::S
    likelihood::L
    density_fn::D
end

function ExtractedCallableStrategy(extractor, sampler;
                                   likelihood=nothing,
                                   density_fn=nothing)
    return ExtractedCallableStrategy{
        typeof(extractor), typeof(sampler), typeof(likelihood), typeof(density_fn)
    }(extractor, sampler, likelihood, density_fn)
end

function StrategyInterface.sample_action(strategy::ExtractedCallableStrategy,
                                         record,
                                         rng::AbstractRNG = Random.default_rng())
    x = strategy.extractor(record)
    return strategy.sampler(x, rng)
end

function StrategyInterface.action_probability(strategy::ExtractedCallableStrategy, record, action)
    strategy.likelihood === nothing &&
        throw(MethodError(StrategyInterface.action_probability, (strategy, record, action)))
    x = strategy.extractor(record)
    return strategy.likelihood(x, action)
end

function StrategyInterface.action_density(strategy::ExtractedCallableStrategy, record, action)
    strategy.density_fn === nothing &&
        throw(MethodError(StrategyInterface.action_density, (strategy, record, action)))
    x = strategy.extractor(record)
    return strategy.density_fn(x, action)
end

end