module DecisionRuleAdapters

using ..Strategies
using ..DecisionRulesInterface
using ..BuiltinDecisionRules

export as_decision_rule
export as_unconditioned_rule
export as_observation_rule
export as_state_rule
export RULE_NONE, RULE_OBSERVATION, RULE_STATE

const RULE_NONE = :none
const RULE_OBSERVATION = :observation
const RULE_STATE = :state

"""
Identity adapter for already-decision-rule objects.
"""
as_decision_rule(rule::DecisionRulesInterface.AbstractDecisionRule) = rule

# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

@inline function _require_applicable_probability(msg::AbstractString, args...)
    applicable(Strategies.probability, args...) || throw(ArgumentError(msg))
    return nothing
end

@inline function _require_nonempty_support(strategy::Strategies.AbstractStrategy, who::AbstractString)
    A = Strategies.support(strategy)
    isempty(A) && throw(ArgumentError("$who requires a strategy with nonempty support."))
    return A
end

# ----------------------------------------------------------------------
# Explicit adapters
# ----------------------------------------------------------------------

"""
Explicitly adapt a compatible strategy to an unconditioned decision rule.

This requires:
- `Strategies.sample_action(strategy, rng)`

Likelihood support is included only when requested explicitly via
`likelihood=:from_strategy` or by passing a custom likelihood function.

When `likelihood=:from_strategy` is requested, this adapter validates eagerly that:
- `Strategies.support(strategy)` exists and is nonempty
- `Strategies.probability(strategy, action)` is applicable for a representative action
"""
function as_unconditioned_rule(strategy::Strategies.AbstractStrategy;
                               likelihood=nothing,
                               memory::DecisionRulesInterface.AbstractMemoryClass = DecisionRulesInterface.Memoryless())
    sampler = rng -> Strategies.sample_action(strategy, rng)

    if likelihood === :from_strategy
        A = _require_nonempty_support(
            strategy,
            "likelihood=:from_strategy for an unconditioned decision rule"
        )
        _require_applicable_probability(
            "likelihood=:from_strategy requires Strategies.probability(strategy, action).",
            strategy, first(A)
        )
        likelihood_fn = action -> Strategies.probability(strategy, action)

    elseif likelihood === nothing
        likelihood_fn = nothing

    else
        likelihood_fn = likelihood
    end

    return BuiltinDecisionRules.UnconditionedDecisionRule(
        sampler; likelihood=likelihood_fn, memory=memory
    )
end

"""
Explicitly adapt a compatible strategy to an observation-conditioned decision rule.

This requires:
- `Strategies.sample_action(strategy, observation, rng)`

Likelihood support is included only when requested explicitly via
`likelihood=:from_strategy` or by passing a custom likelihood function.

Observation-conditioned probability support cannot generally be validated eagerly
without a representative observation. So when `likelihood=:from_strategy` is used,
the returned decision rule delegates likelihood queries to:
- `Strategies.probability(strategy, observation, action)`

and any incompatibility is reported when that likelihood is queried.
"""
function as_observation_rule(strategy::Strategies.AbstractStrategy;
                             likelihood=nothing,
                             memory::DecisionRulesInterface.AbstractMemoryClass = DecisionRulesInterface.Memoryless())
    sampler = (observation, rng) -> Strategies.sample_action(strategy, observation, rng)

    if likelihood === :from_strategy
        likelihood_fn = (observation, action) ->
            Strategies.probability(strategy, observation, action)

    elseif likelihood === nothing
        likelihood_fn = nothing

    else
        likelihood_fn = likelihood
    end

    return BuiltinDecisionRules.ObservationDecisionRule(
        sampler; likelihood=likelihood_fn, memory=memory
    )
end

"""
Explicitly adapt a compatible strategy to a state-conditioned decision rule.

This requires:
- `Strategies.sample_action(strategy, state, rng)`

Likelihood support is included only when requested explicitly via
`likelihood=:from_strategy` or by passing a custom likelihood function.

State-conditioned probability support cannot generally be validated eagerly
without a representative state. So when `likelihood=:from_strategy` is used,
the returned decision rule delegates likelihood queries to:
- `Strategies.probability(strategy, state, action)`

and any incompatibility is reported when that likelihood is queried.
"""
function as_state_rule(strategy::Strategies.AbstractStrategy;
                       likelihood=nothing,
                       memory::DecisionRulesInterface.AbstractMemoryClass = DecisionRulesInterface.Memoryless())
    sampler = (state, rng) -> Strategies.sample_action(strategy, state, rng)

    if likelihood === :from_strategy
        likelihood_fn = (state, action) ->
            Strategies.probability(strategy, state, action)

    elseif likelihood === nothing
        likelihood_fn = nothing

    else
        likelihood_fn = likelihood
    end

    return BuiltinDecisionRules.StateDecisionRule(
        sampler; likelihood=likelihood_fn, memory=memory
    )
end

"""
Explicit convenience adapter for strategies.

This is a front door over the more explicit:
- `as_unconditioned_rule`
- `as_observation_rule`
- `as_state_rule`
"""
function as_decision_rule(strategy::Strategies.AbstractStrategy;
                          conditioning::Symbol = RULE_NONE,
                          likelihood=nothing,
                          memory::DecisionRulesInterface.AbstractMemoryClass = DecisionRulesInterface.Memoryless())
    if conditioning === RULE_NONE
        return as_unconditioned_rule(
            strategy; likelihood=likelihood, memory=memory
        )
    elseif conditioning === RULE_OBSERVATION
        return as_observation_rule(
            strategy; likelihood=likelihood, memory=memory
        )
    elseif conditioning === RULE_STATE
        return as_state_rule(
            strategy; likelihood=likelihood, memory=memory
        )
    else
        throw(ArgumentError(
            "Unknown conditioning=$(conditioning). Expected :none, :observation, or :state."
        ))
    end
end

end