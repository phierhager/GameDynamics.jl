module DecisionRulesInterface

using Random

export AbstractDecisionRule

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
Stable public root for runtime-queryable decision rules.

A decision rule is the library's unified abstraction for what many RL users call a
policy and many game theorists call a strategy.

Decision rules are the action-selection objects queried by simulation, rollout,
runtime, and evaluation layers.
"""
abstract type AbstractDecisionRule end

# ----------------------------------------------------------------------
# Public trait axes
# ----------------------------------------------------------------------

"""
Trait axis describing what object a decision rule is conditioned on at query time.
"""
abstract type AbstractConditioningKind end

"""
Decision rule takes no conditioning input at query time.
"""
struct NoConditioning <: AbstractConditioningKind end

"""
Decision rule is queried on the current observation.
"""
struct ObservationConditioning <: AbstractConditioningKind end

"""
Decision rule is queried on the true current underlying state.

This is a runtime/control abstraction and may be used even when the state is not
observable to the acting player.
"""
struct StateConditioning <: AbstractConditioningKind end

"""
Decision rule is queried on a history object.
"""
struct HistoryConditioning <: AbstractConditioningKind end

"""
Trait axis describing how a decision rule represents dependence on the past.
"""
abstract type AbstractMemoryClass end

"""
Decision rule is memoryless relative to its declared conditioning object.
"""
struct Memoryless <: AbstractMemoryClass end

"""
Decision rule may depend on history in an unrestricted way.
"""
struct HistoryDependent <: AbstractMemoryClass end

"""
Reserved stable extension point for finite-memory decision rules.
"""
struct FiniteMemory <: AbstractMemoryClass end

# ----------------------------------------------------------------------
# Required public trait declarations
# ----------------------------------------------------------------------

"""
Return the declared conditioning kind for a decision-rule type.

Concrete public decision-rule types are expected to implement this explicitly.
"""
function conditioning_kind(::Type{<:AbstractDecisionRule})
    throw(MethodError(conditioning_kind, (AbstractDecisionRule,)))
end

conditioning_kind(rule::AbstractDecisionRule) = conditioning_kind(typeof(rule))

"""
Return the declared memory class for a decision-rule type.

Concrete public decision-rule types are expected to implement this explicitly.
"""
function memory_class(::Type{<:AbstractDecisionRule})
    throw(MethodError(memory_class, (AbstractDecisionRule,)))
end

memory_class(rule::AbstractDecisionRule) = memory_class(typeof(rule))

"""
Whether `action_likelihood` is implemented for this decision-rule type.

When true, `action_likelihood` must return a normalized probabilistic quantity
conditional on the queried input.
"""
has_action_likelihood(::Type{<:AbstractDecisionRule}) = false
has_action_likelihood(rule::AbstractDecisionRule) = has_action_likelihood(typeof(rule))

# ----------------------------------------------------------------------
# Canonical query interface
# ----------------------------------------------------------------------

"""
Sample an action from the decision rule.

Expected signatures by conditioning kind:

- `NoConditioning`:
    `sample_action(rule, rng)`
- `ObservationConditioning`:
    `sample_action(rule, observation, rng)`
- `StateConditioning`:
    `sample_action(rule, state, rng)`
- `HistoryConditioning`:
    `sample_action(rule, history, rng)`

Convenience methods may default `rng` to `Random.default_rng()`.
"""
function sample_action end

sample_action(rule::AbstractDecisionRule) =
    sample_action(rule, Random.default_rng())

sample_action(rule::AbstractDecisionRule, conditioning) =
    sample_action(rule, conditioning, Random.default_rng())

"""
Return the conditional action likelihood for the queried input.

Expected signatures by conditioning kind:

- `NoConditioning`:
    `action_likelihood(rule, action)`
- `ObservationConditioning`:
    `action_likelihood(rule, observation, action)`
- `StateConditioning`:
    `action_likelihood(rule, state, action)`
- `HistoryConditioning`:
    `action_likelihood(rule, history, action)`

This is an optional capability.
"""
function action_likelihood end

function action_likelihood(rule::AbstractDecisionRule, args...)
    has_action_likelihood(rule) || throw(MethodError(action_likelihood, (rule, args...)))
    throw(MethodError(action_likelihood, (rule, args...)))
end

end