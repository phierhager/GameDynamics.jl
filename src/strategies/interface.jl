module StrategyInterface

using Random

export AbstractStrategy

export AbstractContextKind
export NoContext
export StateContext
export ObservationContext
export HistoryContext
export InfosetContext
export CustomContext

export AbstractInternalStateClass
export Stateless
export Stateful
export FiniteStateController

export context_kind
export internal_state_class

export support
export probabilities
export sample_action
export action_probability
export action_density
export local_strategy

export expected_value
export monte_carlo_expectation

"""
Stable public root for strategies.

A strategy is the library's unified action-selection abstraction.
In reinforcement learning, this corresponds to what is often called a policy.

This subsystem is intentionally representation-focused:
- it describes how strategies are represented and queried
- it does not define where contexts come from
- it does not define how internal controller state is threaded across time

Those runtime responsibilities belong elsewhere.
"""
abstract type AbstractStrategy end

# ----------------------------------------------------------------------
# Context kinds
# ----------------------------------------------------------------------

"""
Trait axis describing what kind of context a strategy expects at query time.
"""
abstract type AbstractContextKind end

"""
Strategy takes no context at query time.
"""
struct NoContext <: AbstractContextKind end

"""
Strategy is queried on the true underlying state.
"""
struct StateContext <: AbstractContextKind end

"""
Strategy is queried on the current observation.
"""
struct ObservationContext <: AbstractContextKind end

"""
Strategy is queried on a history object.
"""
struct HistoryContext <: AbstractContextKind end

"""
Strategy is queried on an infoset-like object.
"""
struct InfosetContext <: AbstractContextKind end

"""
Strategy is queried on a user-defined custom context object.
"""
struct CustomContext <: AbstractContextKind end

# ----------------------------------------------------------------------
# Internal controller-state classes
# ----------------------------------------------------------------------

"""
Trait axis describing whether the strategy maintains internal controller state beyond
the supplied query context.

This is currently descriptive metadata, not an execution protocol.
"""
abstract type AbstractInternalStateClass end

"""
Strategy has no internal controller state beyond the supplied query context.
"""
struct Stateless <: AbstractInternalStateClass end

"""
Strategy may maintain unrestricted internal controller state.
"""
struct Stateful <: AbstractInternalStateClass end

"""
Strategy maintains finite internal controller state.

This is currently descriptive metadata, not a concrete controller API.
"""
struct FiniteStateController <: AbstractInternalStateClass end

# ----------------------------------------------------------------------
# Trait declarations
# ----------------------------------------------------------------------

"""
Return the declared context kind for a strategy type.
"""
function context_kind(::Type{<:AbstractStrategy})
    throw(MethodError(context_kind, (AbstractStrategy,)))
end

context_kind(strategy::AbstractStrategy) = context_kind(typeof(strategy))

"""
Return the declared internal-state class for a strategy type.
"""
function internal_state_class(::Type{<:AbstractStrategy})
    throw(MethodError(internal_state_class, (AbstractStrategy,)))
end

internal_state_class(strategy::AbstractStrategy) = internal_state_class(typeof(strategy))

# ----------------------------------------------------------------------
# Capability interfaces
# ----------------------------------------------------------------------

"""
Return the finite support of a strategy, when such a notion is available.

For some non-enumerated strategies this may instead denote an admissible domain.
"""
function support end

"""
Return probabilities aligned with `support(strategy)`, when such a notion is available.
"""
function probabilities end

"""
Sample an action from the strategy.

Expected signatures include:
- `sample_action(strategy, rng)` for `NoContext` strategies
- `sample_action(strategy, context, rng)` for contextual strategies
"""
function sample_action end

sample_action(strategy::AbstractStrategy) =
    sample_action(strategy, Random.default_rng())

sample_action(strategy::AbstractStrategy, context) =
    sample_action(strategy, context, Random.default_rng())

"""
Return the action probability for the queried input.

Examples:
- `action_probability(strategy, action)`
- `action_probability(strategy, context, action)`
"""
function action_probability end

"""
Return the action density for the queried input, when density is defined.

Examples:
- `action_density(strategy, action)`
- `action_density(strategy, context, action)`
"""
function action_density end

"""
Return a local strategy conditioned on some indexing object, when the strategy is a
container of local strategies.
"""
function local_strategy end

# ----------------------------------------------------------------------
# Generic evaluation helpers
# ----------------------------------------------------------------------

"""
Expected value under a finite-support strategy.

`values` must align with `support(strategy)`.
"""
function expected_value(strategy::AbstractStrategy, values)
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
Monte Carlo expectation under a sampleable strategy.
"""
function monte_carlo_expectation(f,
                                 strategy::AbstractStrategy;
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