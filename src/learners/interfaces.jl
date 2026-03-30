module LearningInterfaces

using Random
using ..RuntimeRecords

export AbstractLearner
export AbstractLearnerState
export AbstractLearningContext
export AbstractActionDistribution
export action_mode
export reset!
export act!
export update!
export policy!
export requires_feedback_type
export supports_action_space
export learner_name

abstract type AbstractLearner end
abstract type AbstractLearnerState end
abstract type AbstractLearningContext end
abstract type AbstractActionDistribution end

"""
Learner action interface classification.

Suggested meanings:
- `:discrete_index`  => returns integer action ids
- `:structured`      => returns typed action objects
- `:distribution`    => writes a policy/distribution into a buffer or object
- `:parametric`      => learner acts through continuous parameters / policy object
"""
action_mode(::AbstractLearner) = :discrete_index

"""
Optional human-readable learner name.
"""
learner_name(l::AbstractLearner) = nameof(typeof(l))

"""
Reset mutable learner state in place.
"""
function reset!(learner::AbstractLearner, state::AbstractLearnerState)
    throw(MethodError(reset!, (learner, state)))
end

"""
Sample or choose an action from the learner under context `ctx`.

Contract:
- should be allocation-light
- should return a valid action for the learner/environment pair
"""
function act!(learner::AbstractLearner,
              state::AbstractLearnerState,
              ctx::AbstractLearningContext,
              rng::AbstractRNG = Random.default_rng())
    throw(MethodError(act!, (learner, state, ctx, rng)))
end

"""
Update learner state from a runtime record.
"""
function update!(learner::AbstractLearner,
                 state::AbstractLearnerState,
                 record::RuntimeRecords.AbstractStepRecord)
    throw(MethodError(update!, (learner, state, record)))
end

"""
Write the learner's current policy / action distribution into `dest`.

This is preferred over returning newly allocated vectors in hot loops.
"""
function policy!(dest,
                 learner::AbstractLearner,
                 state::AbstractLearnerState,
                 ctx::AbstractLearningContext)
    throw(MethodError(policy!, (dest, learner, state, ctx)))
end

"""
Declare the preferred runtime-record type for this learner family.
"""
requires_feedback_type(::AbstractLearner) = RuntimeRecords.AbstractStepRecord

"""
Optional capability hook describing which action-space category the learner expects.

Examples:
- `:finite_discrete`
- `:continuous_box`
- `:simplex`
"""
supports_action_space(::AbstractLearner) = :unknown

"""
Fallback allocation-friendly helper for policy extraction.

Only intended for diagnostics / inspection. Hot paths should use `policy!`.
"""
function Base.copy(learner_state_pair::Tuple{AbstractLearner,AbstractLearnerState,AbstractLearningContext})
    learner, state, ctx = learner_state_pair
    throw(ArgumentError("No generic allocation policy helper is defined for $(typeof(learner)). Implement `policy!` for high-performance use."))
end

end