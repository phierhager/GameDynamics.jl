module PolicyAdapters

using ..Strategies
using ..PolicyCore
using ..PolicyTypes

export as_policy
export as_unconditioned_policy
export as_observation_policy
export as_state_policy
export POLICY_NONE, POLICY_OBSERVATION, POLICY_STATE

const POLICY_NONE = :none
const POLICY_OBSERVATION = :observation
const POLICY_STATE = :state

"""
Identity adapter for already-policy objects.
"""
as_policy(policy::PolicyCore.AbstractPolicy) = policy

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
Explicitly adapt a compatible strategy to an unconditioned runtime policy.

This requires:
- `Strategies.sample_action(strategy, rng)`

Likelihood support is included only when requested explicitly via
`likelihood=:from_strategy` or by passing a custom likelihood function.

When `likelihood=:from_strategy` is requested, this adapter validates eagerly that:
- `Strategies.support(strategy)` exists and is nonempty
- `Strategies.probability(strategy, action)` is applicable for a representative action
"""
function as_unconditioned_policy(strategy::Strategies.AbstractStrategy;
                                 likelihood=nothing,
                                 memory::PolicyCore.AbstractMemoryClass = PolicyCore.Memoryless())
    sampler = rng -> Strategies.sample_action(strategy, rng)

    if likelihood === :from_strategy
        A = _require_nonempty_support(
            strategy,
            "likelihood=:from_strategy for an unconditioned policy"
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

    return PolicyTypes.UnconditionedPolicy(
        sampler; likelihood=likelihood_fn, memory=memory
    )
end

"""
Explicitly adapt a compatible strategy to an observation-conditioned runtime policy.

This requires:
- `Strategies.sample_action(strategy, observation, rng)`

Likelihood support is included only when requested explicitly via
`likelihood=:from_strategy` or by passing a custom likelihood function.

Observation-conditioned probability support cannot generally be validated eagerly
without a representative observation. So when `likelihood=:from_strategy` is used,
the returned policy delegates likelihood queries to:
- `Strategies.probability(strategy, observation, action)`

and any incompatibility is reported when that likelihood is queried.
"""
function as_observation_policy(strategy::Strategies.AbstractStrategy;
                               likelihood=nothing,
                               memory::PolicyCore.AbstractMemoryClass = PolicyCore.Memoryless())
    sampler = (observation, rng) -> Strategies.sample_action(strategy, observation, rng)

    if likelihood === :from_strategy
        likelihood_fn = (observation, action) ->
            Strategies.probability(strategy, observation, action)

    elseif likelihood === nothing
        likelihood_fn = nothing

    else
        likelihood_fn = likelihood
    end

    return PolicyTypes.ObservationPolicy(
        sampler; likelihood=likelihood_fn, memory=memory
    )
end

"""
Explicitly adapt a compatible strategy to a state-conditioned runtime policy.

This requires:
- `Strategies.sample_action(strategy, state, rng)`

Likelihood support is included only when requested explicitly via
`likelihood=:from_strategy` or by passing a custom likelihood function.

State-conditioned probability support cannot generally be validated eagerly
without a representative state. So when `likelihood=:from_strategy` is used,
the returned policy delegates likelihood queries to:
- `Strategies.probability(strategy, state, action)`

and any incompatibility is reported when that likelihood is queried.
"""
function as_state_policy(strategy::Strategies.AbstractStrategy;
                         likelihood=nothing,
                         memory::PolicyCore.AbstractMemoryClass = PolicyCore.Memoryless())
    sampler = (state, rng) -> Strategies.sample_action(strategy, state, rng)

    if likelihood === :from_strategy
        likelihood_fn = (state, action) ->
            Strategies.probability(strategy, state, action)

    elseif likelihood === nothing
        likelihood_fn = nothing

    else
        likelihood_fn = likelihood
    end

    return PolicyTypes.StatePolicy(
        sampler; likelihood=likelihood_fn, memory=memory
    )
end

"""
Explicit convenience adapter for strategies.

This is a front door over the more explicit:
- `as_unconditioned_policy`
- `as_observation_policy`
- `as_state_policy`
"""
function as_policy(strategy::Strategies.AbstractStrategy;
                   conditioning::Symbol = POLICY_NONE,
                   likelihood=nothing,
                   memory::PolicyCore.AbstractMemoryClass = PolicyCore.Memoryless())
    if conditioning === POLICY_NONE
        return as_unconditioned_policy(
            strategy; likelihood=likelihood, memory=memory
        )
    elseif conditioning === POLICY_OBSERVATION
        return as_observation_policy(
            strategy; likelihood=likelihood, memory=memory
        )
    elseif conditioning === POLICY_STATE
        return as_state_policy(
            strategy; likelihood=likelihood, memory=memory
        )
    else
        throw(ArgumentError(
            "Unknown conditioning=$(conditioning). Expected :none, :observation, or :state."
        ))
    end
end

end