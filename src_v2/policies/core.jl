module PolicyCore

using Random

export AbstractPolicy

export AbstractConditioningKind
export NoConditioning
export ObservationConditioning
export StateConditioning
export HistoryConditioning

export AbstractMemoryClass
export Memoryless
export HistoryDependent
export FiniteMemory

export conditioning_kind
export memory_class
export has_action_likelihood

export sample_action
export action_likelihood

"""
Stable public root for runtime-queryable policies.

`AbstractPolicy` is intentionally distinct from theory-layer strategy objects.
Policies are the execution-time control abstraction queried by simulation,
rollout, and solver/runtime layers.
"""
abstract type AbstractPolicy end

# ----------------------------------------------------------------------
# Public trait axes
# ----------------------------------------------------------------------

"""
Trait axis describing what object a policy is conditioned on at query time.
"""
abstract type AbstractConditioningKind end

"""
Policy takes no conditioning input at query time.
"""
struct NoConditioning <: AbstractConditioningKind end

"""
Policy is queried on the current observation.
"""
struct ObservationConditioning <: AbstractConditioningKind end

"""
Policy is queried on the true current underlying state.

This is a runtime/control abstraction and may be used even when the state is
not observable to the acting player.
"""
struct StateConditioning <: AbstractConditioningKind end

"""
Policy is queried on a history object.
"""
struct HistoryConditioning <: AbstractConditioningKind end

"""
Trait axis describing how a policy represents dependence on the past.
"""
abstract type AbstractMemoryClass end

"""
Policy is memoryless relative to its declared conditioning object.
"""
struct Memoryless <: AbstractMemoryClass end

"""
Policy may depend on history in an unrestricted way.
"""
struct HistoryDependent <: AbstractMemoryClass end

"""
Reserved stable extension point for finite-memory policies.
"""
struct FiniteMemory <: AbstractMemoryClass end

# ----------------------------------------------------------------------
# Required public trait declarations
# ----------------------------------------------------------------------

"""
Return the declared conditioning kind for a policy type.

Concrete public policy types are expected to implement this explicitly.
"""
function conditioning_kind(::Type{<:AbstractPolicy})
    throw(MethodError(conditioning_kind, (AbstractPolicy,)))
end

conditioning_kind(policy::AbstractPolicy) = conditioning_kind(typeof(policy))

"""
Return the declared memory class for a policy type.

Concrete public policy types are expected to implement this explicitly.
"""
function memory_class(::Type{<:AbstractPolicy})
    throw(MethodError(memory_class, (AbstractPolicy,)))
end

memory_class(policy::AbstractPolicy) = memory_class(typeof(policy))

"""
Whether `action_likelihood` is implemented for this policy type.

When true, `action_likelihood` must return a normalized probabilistic quantity
conditional on the queried input.
"""
has_action_likelihood(::Type{<:AbstractPolicy}) = false
has_action_likelihood(policy::AbstractPolicy) = has_action_likelihood(typeof(policy))

# ----------------------------------------------------------------------
# Canonical query interface
# ----------------------------------------------------------------------

"""
Sample an action from the policy.

Expected signatures by conditioning kind:

- `NoConditioning`:
    `sample_action(policy, rng)`
- `ObservationConditioning`:
    `sample_action(policy, observation, rng)`
- `StateConditioning`:
    `sample_action(policy, state, rng)`
- `HistoryConditioning`:
    `sample_action(policy, history, rng)`

Convenience methods may default `rng` to `Random.default_rng()`.
"""
function sample_action end

sample_action(policy::AbstractPolicy) =
    sample_action(policy, Random.default_rng())

sample_action(policy::AbstractPolicy, conditioning) =
    sample_action(policy, conditioning, Random.default_rng())

"""
Return the conditional action likelihood for the queried input.

Expected signatures by conditioning kind:

- `NoConditioning`:
    `action_likelihood(policy, action)`
- `ObservationConditioning`:
    `action_likelihood(policy, observation, action)`
- `StateConditioning`:
    `action_likelihood(policy, state, action)`
- `HistoryConditioning`:
    `action_likelihood(policy, history, action)`

This is an optional capability.
"""
function action_likelihood end

function action_likelihood(policy::AbstractPolicy, args...)
    has_action_likelihood(policy) || throw(MethodError(action_likelihood, (policy, args...)))
    throw(MethodError(action_likelihood, (policy, args...)))
end

end