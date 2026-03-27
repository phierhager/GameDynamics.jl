module DecisionRulesInterface

using Random

export AbstractDecisionRule

export AbstractConditioningKind
export NoConditioning
export ObservationConditioning
export StateConditioning
export HistoryConditioning
export InfosetConditioning

export AbstractMemoryClass
export Memoryless
export HistoryDependent
export FiniteMemory

export conditioning_kind
export memory_class

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
"""
abstract type AbstractDecisionRule end

# ----------------------------------------------------------------------
# Conditioning kinds
# ----------------------------------------------------------------------

abstract type AbstractConditioningKind end

"""
Rule takes no conditioning input at query time.
"""
struct NoConditioning <: AbstractConditioningKind end

"""
Rule is queried on the current observation.
"""
struct ObservationConditioning <: AbstractConditioningKind end

"""
Rule is queried on the true underlying state.
"""
struct StateConditioning <: AbstractConditioningKind end

"""
Rule is queried on a history object.
"""
struct HistoryConditioning <: AbstractConditioningKind end

"""
Rule is queried on an infoset-like object.
"""
struct InfosetConditioning <: AbstractConditioningKind end

# ----------------------------------------------------------------------
# Memory classes
# ----------------------------------------------------------------------

abstract type AbstractMemoryClass end

"""
Rule is memoryless relative to its declared conditioning object.
"""
struct Memoryless <: AbstractMemoryClass end

"""
Rule may depend on history in an unrestricted way.
"""
struct HistoryDependent <: AbstractMemoryClass end

"""
Reserved stable extension point for finite-memory rules.
"""
struct FiniteMemory <: AbstractMemoryClass end

# ----------------------------------------------------------------------
# Trait declarations
# ----------------------------------------------------------------------

"""
Return the declared conditioning kind for a decision-rule type.
"""
function conditioning_kind(::Type{<:AbstractDecisionRule})
    throw(MethodError(conditioning_kind, (AbstractDecisionRule,)))
end

conditioning_kind(rule::AbstractDecisionRule) = conditioning_kind(typeof(rule))

"""
Return the declared memory class for a decision-rule type.
"""
function memory_class(::Type{<:AbstractDecisionRule})
    throw(MethodError(memory_class, (AbstractDecisionRule,)))
end

memory_class(rule::AbstractDecisionRule) = memory_class(typeof(rule))

# ----------------------------------------------------------------------
# Capability interfaces
# ----------------------------------------------------------------------

"""
Return the finite support of a rule, when such a notion is available.
"""
function support end

"""
Return probabilities aligned with `support(rule)`, when such a notion is available.
"""
function probabilities end

"""
Sample an action from the rule.

Expected signatures include:
- `sample_action(rule, rng)` for unconditioned rules
- `sample_action(rule, conditioning, rng)` for conditioned rules
"""
function sample_action end

sample_action(rule::AbstractDecisionRule) =
    sample_action(rule, Random.default_rng())

sample_action(rule::AbstractDecisionRule, conditioning) =
    sample_action(rule, conditioning, Random.default_rng())

"""
Return the action probability for the queried input.

Examples:
- `action_probability(rule, action)`
- `action_probability(rule, observation, action)`
- `action_probability(rule, infoset, action)`
"""
function action_probability end

"""
Return the action density for the queried input, when density is defined.

Examples:
- `action_density(rule, action)`
- `action_density(rule, state, action)`
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