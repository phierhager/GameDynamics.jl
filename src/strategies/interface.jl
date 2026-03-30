module StrategyInterface

using Random

export AbstractStrategy
export AbstractLocalStrategy
export AbstractRecordStrategy

export support
export probabilities
export sample_action
export action_probability
export action_density

export expected_value
export monte_carlo_expectation

# ----------------------------------------------------------------------
# Abstract roots
# ----------------------------------------------------------------------

"""
Root of all strategies.
"""
abstract type AbstractStrategy end

"""
A strategy that can be queried without a runtime record.
Examples:
- deterministic local action rules
- finite mixed actions
- continuous action samplers
"""
abstract type AbstractLocalStrategy <: AbstractStrategy end

"""
A strategy that is queried with a runtime record.
Examples:
- direct callable policies on records
- extractor-based wrappers around local strategies
"""
abstract type AbstractRecordStrategy <: AbstractStrategy end

# ----------------------------------------------------------------------
# Capability interfaces
# ----------------------------------------------------------------------

function support end
function probabilities end

"""
Sample an action.

Expected signatures:
- `sample_action(strategy, rng)` for local strategies
- `sample_action(strategy, record, rng)` for record strategies
"""
function sample_action end

sample_action(strategy::AbstractStrategy) =
    sample_action(strategy, Random.default_rng())

sample_action(strategy::AbstractRecordStrategy, record) =
    sample_action(strategy, record, Random.default_rng())

function action_probability end
function action_density end

# ----------------------------------------------------------------------
# Generic evaluation helpers
# ----------------------------------------------------------------------

"""
Expected value under a finite-support local strategy.
"""
function expected_value(strategy::AbstractLocalStrategy, values)
    ps = probabilities(strategy)
    length(values) == length(ps) ||
        throw(ArgumentError("Values must align with the strategy support."))
    acc = 0.0
    @inbounds for i in eachindex(ps)
        acc += ps[i] * values[i]
    end
    return acc
end

"""
Monte Carlo expectation under a sampleable local strategy.
"""
function monte_carlo_expectation(f,
                                 strategy::AbstractLocalStrategy;
                                 rng::AbstractRNG = Random.default_rng(),
                                 n_samples::Int = 1024)
    n_samples > 0 || throw(ArgumentError("n_samples must be positive."))
    acc = 0.0
    for _ in 1:n_samples
        acc += f(sample_action(strategy, rng))
    end
    return acc / n_samples
end

end