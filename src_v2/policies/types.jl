module PolicyTypes

using Random
using ..PolicyCore

export UnconditionedPolicy
export ObservationPolicy
export StatePolicy

"""
Shared root for built-in native runtime policy types.
"""
abstract type AbstractNativePolicy <: PolicyCore.AbstractPolicy end

"""
Native runtime policy with no conditioning input at query time.

This means the policy is queried as `sample_action(policy, rng)`, not that the
world lacks state.
"""
struct UnconditionedPolicy{S,L,M<:PolicyCore.AbstractMemoryClass} <: AbstractNativePolicy
    sampler::S
    likelihood::L
    memory::M
end

"""
Native runtime policy queried on the current observation.
"""
struct ObservationPolicy{S,L,M<:PolicyCore.AbstractMemoryClass} <: AbstractNativePolicy
    sampler::S
    likelihood::L
    memory::M
end

"""
Native runtime policy queried on the true current underlying state.
"""
struct StatePolicy{S,L,M<:PolicyCore.AbstractMemoryClass} <: AbstractNativePolicy
    sampler::S
    likelihood::L
    memory::M
end

# ----------------------------------------------------------------------
# Constructors
# ----------------------------------------------------------------------

"""
Create an unconditioned runtime policy.

Expected sampler signature:
- `sampler(rng)`

Optional likelihood signature:
- `likelihood(action)`
"""
UnconditionedPolicy(sampler;
                    likelihood=nothing,
                    memory::PolicyCore.AbstractMemoryClass=PolicyCore.Memoryless()) =
    UnconditionedPolicy{typeof(sampler),typeof(likelihood),typeof(memory)}(
        sampler, likelihood, memory
    )

"""
Create an observation-conditioned runtime policy.

Expected sampler signature:
- `sampler(observation, rng)`

Optional likelihood signature:
- `likelihood(observation, action)`
"""
ObservationPolicy(sampler;
                  likelihood=nothing,
                  memory::PolicyCore.AbstractMemoryClass=PolicyCore.Memoryless()) =
    ObservationPolicy{typeof(sampler),typeof(likelihood),typeof(memory)}(
        sampler, likelihood, memory
    )

"""
Create a state-conditioned runtime policy.

Expected sampler signature:
- `sampler(state, rng)`

Optional likelihood signature:
- `likelihood(state, action)`
"""
StatePolicy(sampler;
            likelihood=nothing,
            memory::PolicyCore.AbstractMemoryClass=PolicyCore.Memoryless()) =
    StatePolicy{typeof(sampler),typeof(likelihood),typeof(memory)}(
        sampler, likelihood, memory
    )

# ----------------------------------------------------------------------
# Trait declarations
# ----------------------------------------------------------------------

PolicyCore.conditioning_kind(::Type{<:UnconditionedPolicy}) = PolicyCore.NoConditioning()
PolicyCore.conditioning_kind(::Type{<:ObservationPolicy}) = PolicyCore.ObservationConditioning()
PolicyCore.conditioning_kind(::Type{<:StatePolicy}) = PolicyCore.StateConditioning()

PolicyCore.memory_class(::Type{<:UnconditionedPolicy{S,L,M}}) where {S,L,M<:PolicyCore.AbstractMemoryClass} = M()
PolicyCore.memory_class(::Type{<:ObservationPolicy{S,L,M}}) where {S,L,M<:PolicyCore.AbstractMemoryClass} = M()
PolicyCore.memory_class(::Type{<:StatePolicy{S,L,M}}) where {S,L,M<:PolicyCore.AbstractMemoryClass} = M()

PolicyCore.has_action_likelihood(::Type{<:UnconditionedPolicy{S,L}}) where {S,L} = !(L <: Nothing)
PolicyCore.has_action_likelihood(::Type{<:ObservationPolicy{S,L}}) where {S,L} = !(L <: Nothing)
PolicyCore.has_action_likelihood(::Type{<:StatePolicy{S,L}}) where {S,L} = !(L <: Nothing)

# ----------------------------------------------------------------------
# Canonical query implementations
# ----------------------------------------------------------------------

PolicyCore.sample_action(policy::UnconditionedPolicy,
                         rng::AbstractRNG = Random.default_rng()) =
    policy.sampler(rng)

PolicyCore.sample_action(policy::ObservationPolicy,
                         observation,
                         rng::AbstractRNG = Random.default_rng()) =
    policy.sampler(observation, rng)

PolicyCore.sample_action(policy::StatePolicy,
                         state,
                         rng::AbstractRNG = Random.default_rng()) =
    policy.sampler(state, rng)

function PolicyCore.action_likelihood(policy::UnconditionedPolicy, action)
    PolicyCore.has_action_likelihood(policy) || throw(MethodError(PolicyCore.action_likelihood, (policy, action)))
    return policy.likelihood(action)
end

function PolicyCore.action_likelihood(policy::ObservationPolicy, observation, action)
    PolicyCore.has_action_likelihood(policy) || throw(MethodError(PolicyCore.action_likelihood, (policy, observation, action)))
    return policy.likelihood(observation, action)
end

function PolicyCore.action_likelihood(policy::StatePolicy, state, action)
    PolicyCore.has_action_likelihood(policy) || throw(MethodError(PolicyCore.action_likelihood, (policy, state, action)))
    return policy.likelihood(state, action)
end

end