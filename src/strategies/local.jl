module LocalStrategies

using Random
using ..StrategyInterface
using ..StrategyInternalUtils

export DeterministicStrategy
export FiniteMixedStrategy

export ContextualStrategy
export unconditioned_strategy
export observation_strategy
export state_strategy
export history_strategy
export infoset_strategy
export custom_context_strategy

export SamplerStrategy
export SamplerDensityStrategy

# ----------------------------------------------------------------------
# Deterministic / finite mixed
# ----------------------------------------------------------------------

struct DeterministicStrategy{A} <: StrategyInterface.AbstractStrategy
    action::A
end

StrategyInterface.context_kind(::Type{<:DeterministicStrategy}) =
    StrategyInterface.NoContext()

StrategyInterface.internal_state_class(::Type{<:DeterministicStrategy}) =
    StrategyInterface.Stateless()

StrategyInterface.support(strategy::DeterministicStrategy) = (strategy.action,)
StrategyInterface.probabilities(::DeterministicStrategy) = (1.0,)

StrategyInterface.sample_action(strategy::DeterministicStrategy,
                                     rng::AbstractRNG = Random.default_rng()) =
    strategy.action

StrategyInterface.action_probability(strategy::DeterministicStrategy, action) =
    action == strategy.action ? 1.0 : 0.0

struct FiniteMixedStrategy{A,P} <: StrategyInterface.AbstractStrategy
    actions::A
    probs::P

    function FiniteMixedStrategy(actions, probs)
        acts, p = StrategyInternalUtils.canonicalize_support_probs(actions, probs)
        return new{typeof(acts),typeof(p)}(acts, p)
    end
end

FiniteMixedStrategy(probs) =
    FiniteMixedStrategy(Base.OneTo(length(probs)), probs)

StrategyInterface.context_kind(::Type{<:FiniteMixedStrategy}) =
    StrategyInterface.NoContext()

StrategyInterface.internal_state_class(::Type{<:FiniteMixedStrategy}) =
    StrategyInterface.Stateless()

StrategyInterface.support(strategy::FiniteMixedStrategy) = strategy.actions
StrategyInterface.probabilities(strategy::FiniteMixedStrategy) = strategy.probs

function StrategyInterface.sample_action(strategy::FiniteMixedStrategy,
                                              rng::AbstractRNG = Random.default_rng())
    r = rand(rng)
    c = 0.0
    A = StrategyInterface.support(strategy)
    P = StrategyInterface.probabilities(strategy)
    @inbounds for i in eachindex(P)
        c += P[i]
        if r <= c
            return A[i]
        end
    end
    return A[last(eachindex(P))]
end

function StrategyInterface.action_probability(strategy::FiniteMixedStrategy, action)
    A = StrategyInterface.support(strategy)
    P = StrategyInterface.probabilities(strategy)
    @inbounds for i in eachindex(P)
        if A[i] == action
            return Float64(P[i])
        end
    end
    return 0.0
end

# ----------------------------------------------------------------------
# Generic contextual strategy
# ----------------------------------------------------------------------

"""
Generic callable strategy parameterized by its context kind.

Expected signatures:
- `sampler(rng)` when `CK == NoContext`
- `sampler(context, rng)` otherwise

Optional likelihood signatures:
- `likelihood(action)` when `CK == NoContext`
- `likelihood(context, action)` otherwise

This type does not itself verify that the runtime-supplied context matches `CK`;
that compatibility check belongs to the runtime/query layer.
"""
struct ContextualStrategy{
    CK<:StrategyInterface.AbstractContextKind,
    S,
    L,
    ISC<:StrategyInterface.AbstractInternalStateClass,
} <: StrategyInterface.AbstractStrategy
    sampler::S
    likelihood::L
    internal_state::ISC
end

function ContextualStrategy(::Type{CK},
                                sampler;
                                likelihood=nothing,
                                internal_state::StrategyInterface.AbstractInternalStateClass=StrategyInterface.Stateless()) where {CK<:StrategyInterface.AbstractContextKind}
    return ContextualStrategy{CK,typeof(sampler),typeof(likelihood),typeof(internal_state)}(
        sampler, likelihood, internal_state
    )
end

StrategyInterface.context_kind(::Type{<:ContextualStrategy{CK}}) where {CK} = CK()

StrategyInterface.internal_state_class(::Type{<:ContextualStrategy{CK,S,L,ISC}}) where {CK,S,L,ISC<:StrategyInterface.AbstractInternalStateClass} =
    ISC()

function StrategyInterface.sample_action(strategy::ContextualStrategy{StrategyInterface.NoContext},
                                              rng::AbstractRNG = Random.default_rng())
    return strategy.sampler(rng)
end

function StrategyInterface.sample_action(strategy::ContextualStrategy, context, rng::AbstractRNG = Random.default_rng())
    return strategy.sampler(context, rng)
end

function StrategyInterface.action_probability(strategy::ContextualStrategy{StrategyInterface.NoContext}, action)
    strategy.likelihood === nothing &&
        throw(MethodError(StrategyInterface.action_probability, (strategy, action)))
    return strategy.likelihood(action)
end

function StrategyInterface.action_probability(strategy::ContextualStrategy, context, action)
    strategy.likelihood === nothing &&
        throw(MethodError(StrategyInterface.action_probability, (strategy, context, action)))
    return strategy.likelihood(context, action)
end

# Convenience constructors

unconditioned_strategy(sampler;
                   likelihood=nothing,
                   internal_state::StrategyInterface.AbstractInternalStateClass=StrategyInterface.Stateless()) =
    ContextualStrategy(StrategyInterface.NoContext, sampler;
                           likelihood=likelihood,
                           internal_state=internal_state)

observation_strategy(sampler;
                 likelihood=nothing,
                 internal_state::StrategyInterface.AbstractInternalStateClass=StrategyInterface.Stateless()) =
    ContextualStrategy(StrategyInterface.ObservationContext, sampler;
                           likelihood=likelihood,
                           internal_state=internal_state)

state_strategy(sampler;
           likelihood=nothing,
           internal_state::StrategyInterface.AbstractInternalStateClass=StrategyInterface.Stateless()) =
    ContextualStrategy(StrategyInterface.StateContext, sampler;
                           likelihood=likelihood,
                           internal_state=internal_state)

history_strategy(sampler;
             likelihood=nothing,
             internal_state::StrategyInterface.AbstractInternalStateClass=StrategyInterface.Stateless()) =
    ContextualStrategy(StrategyInterface.HistoryContext, sampler;
                           likelihood=likelihood,
                           internal_state=internal_state)

infoset_strategy(sampler;
             likelihood=nothing,
             internal_state::StrategyInterface.AbstractInternalStateClass=StrategyInterface.Stateless()) =
    ContextualStrategy(StrategyInterface.InfosetContext, sampler;
                           likelihood=likelihood,
                           internal_state=internal_state)

custom_context_strategy(sampler;
                    likelihood=nothing,
                    internal_state::StrategyInterface.AbstractInternalStateClass=StrategyInterface.Stateless()) =
    ContextualStrategy(StrategyInterface.CustomContext, sampler;
                           likelihood=likelihood,
                           internal_state=internal_state)

# ----------------------------------------------------------------------
# Continuous sampler-based local strategies
# ----------------------------------------------------------------------

"""
Sampler-only strategy over a continuous or otherwise non-enumerated action set.

`domain` denotes an admissible action set, not necessarily finite support in the
strict discrete sense.
"""
struct SamplerStrategy{S,Dom} <: StrategyInterface.AbstractStrategy
    sampler::S
    domain::Dom
end

"""
Sampler + density strategy over a continuous or otherwise non-enumerated
action set.

`domain` denotes an admissible action set, not necessarily finite support in the
strict discrete sense.
"""
struct SamplerDensityStrategy{S,D,Dom} <: StrategyInterface.AbstractStrategy
    sampler::S
    density_fn::D
    domain::Dom
end

StrategyInterface.context_kind(::Type{<:SamplerStrategy}) =
    StrategyInterface.NoContext()
StrategyInterface.context_kind(::Type{<:SamplerDensityStrategy}) =
    StrategyInterface.NoContext()

StrategyInterface.internal_state_class(::Type{<:SamplerStrategy}) =
    StrategyInterface.Stateless()
StrategyInterface.internal_state_class(::Type{<:SamplerDensityStrategy}) =
    StrategyInterface.Stateless()

StrategyInterface.support(strategy::SamplerStrategy) = strategy.domain
StrategyInterface.support(strategy::SamplerDensityStrategy) = strategy.domain

StrategyInterface.sample_action(strategy::SamplerStrategy,
                                     rng::AbstractRNG = Random.default_rng()) =
    strategy.sampler(rng)

StrategyInterface.sample_action(strategy::SamplerDensityStrategy,
                                     rng::AbstractRNG = Random.default_rng()) =
    strategy.sampler(rng)

StrategyInterface.action_density(strategy::SamplerDensityStrategy, action) =
    strategy.density_fn(action)

end