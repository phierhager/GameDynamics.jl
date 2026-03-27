module DecisionRulesInterface

using Random

export AbstractDecisionRule

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
export local_rule

export expected_value
export monte_carlo_expectation

"""
Stable public root for decision rules.

A decision rule is the library's unified action-selection abstraction. It covers
what RL often calls a policy and what game theory often calls a strategy.

This subsystem is intentionally representation-focused:
- it describes how rules are represented and queried
- it does not define where contexts come from
- it does not define how internal controller state is threaded across time

Those runtime responsibilities belong elsewhere.
"""
abstract type AbstractDecisionRule end

# ----------------------------------------------------------------------
# Context kinds
# ----------------------------------------------------------------------

"""
Trait axis describing what kind of context a decision rule expects at query time.
"""
abstract type AbstractContextKind end

"""
Rule takes no context at query time.
"""
struct NoContext <: AbstractContextKind end

"""
Rule is queried on the true underlying state.
"""
struct StateContext <: AbstractContextKind end

"""
Rule is queried on the current observation.
"""
struct ObservationContext <: AbstractContextKind end

"""
Rule is queried on a history object.
"""
struct HistoryContext <: AbstractContextKind end

"""
Rule is queried on an infoset-like object.
"""
struct InfosetContext <: AbstractContextKind end

"""
Rule is queried on a user-defined custom context object.
"""
struct CustomContext <: AbstractContextKind end

# ----------------------------------------------------------------------
# Internal controller-state classes
# ----------------------------------------------------------------------

"""
Trait axis describing whether the rule maintains internal controller state beyond
the supplied query context.

This is currently descriptive metadata, not an execution protocol.
"""
abstract type AbstractInternalStateClass end

"""
Rule has no internal controller state beyond the supplied query context.
"""
struct Stateless <: AbstractInternalStateClass end

"""
Rule may maintain unrestricted internal controller state.
"""
struct Stateful <: AbstractInternalStateClass end

"""
Rule maintains finite internal controller state.

This is currently descriptive metadata, not a concrete controller API.
"""
struct FiniteStateController <: AbstractInternalStateClass end

# ----------------------------------------------------------------------
# Trait declarations
# ----------------------------------------------------------------------

"""
Return the declared context kind for a decision-rule type.
"""
function context_kind(::Type{<:AbstractDecisionRule})
    throw(MethodError(context_kind, (AbstractDecisionRule,)))
end

context_kind(rule::AbstractDecisionRule) = context_kind(typeof(rule))

"""
Return the declared internal-state class for a decision-rule type.
"""
function internal_state_class(::Type{<:AbstractDecisionRule})
    throw(MethodError(internal_state_class, (AbstractDecisionRule,)))
end

internal_state_class(rule::AbstractDecisionRule) = internal_state_class(typeof(rule))

# ----------------------------------------------------------------------
# Capability interfaces
# ----------------------------------------------------------------------

"""
Return the finite support of a rule, when such a notion is available.

For some non-enumerated rules this may instead denote an admissible domain.
"""
function support end

"""
Return probabilities aligned with `support(rule)`, when such a notion is available.
"""
function probabilities end

"""
Sample an action from the rule.

Expected signatures include:
- `sample_action(rule, rng)` for `NoContext` rules
- `sample_action(rule, context, rng)` for contextual rules
"""
function sample_action end

sample_action(rule::AbstractDecisionRule) =
    sample_action(rule, Random.default_rng())

sample_action(rule::AbstractDecisionRule, context) =
    sample_action(rule, context, Random.default_rng())

"""
Return the action probability for the queried input.

Examples:
- `action_probability(rule, action)`
- `action_probability(rule, context, action)`
"""
function action_probability end

"""
Return the action density for the queried input, when density is defined.

Examples:
- `action_density(rule, action)`
- `action_density(rule, context, action)`
"""
function action_density end

"""
Return a local rule conditioned on some indexing object, when the rule is a
container of local rules.
"""
function local_rule end

# ----------------------------------------------------------------------
# Generic evaluation helpers
# ----------------------------------------------------------------------

"""
Expected value under a finite-support rule.

`values` must align with `support(rule)`.
"""
function expected_value(rule::AbstractDecisionRule, values)
    ps = probabilities(rule)
    length(values) == length(ps) ||
        throw(ArgumentError("Values must align with the decision-rule support."))
    acc = 0.0
    @inbounds for i in eachindex(ps)
        acc += ps[i] * values[i]
    end
    return acc
end

"""
Monte Carlo expectation under a sampleable rule.
"""
function monte_carlo_expectation(f,
                                 rule::AbstractDecisionRule;
                                 rng::AbstractRNG = Random.default_rng(),
                                 n_samples::Int = 1024)
    n_samples > 0 || throw(ArgumentError("n_samples must be positive."))
    acc = 0.0
    for _ in 1:n_samples
        acc += f(sample_action(rule, rng))
    end
    return acc / n_samples
end

end