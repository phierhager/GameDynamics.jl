module LearningInterfaces

using Random
using ..RuntimeRecords

export AbstractLearner
export AbstractLearnerState
export AbstractActionDistribution

export action_mode
export reset!
export act!
export update!
export strategy!
export requires_feedback_type
export supports_action_space
export learner_name

export record_context
export require_context
export has_context

abstract type AbstractLearner end
abstract type AbstractLearnerState end
abstract type AbstractActionDistribution end

action_mode(::AbstractLearner) = :discrete_index
learner_name(l::AbstractLearner) = nameof(typeof(l))

"""
Extract optional context from a runtime record.

Returns `nothing` for non-contextual records.
"""
@inline has_context(rec) = hasproperty(rec, :context)

@inline function record_context(rec::RuntimeRecords.AbstractStepRecord)
    return hasproperty(rec, :context) ? getproperty(rec, :context) : nothing
end

@inline record_context(::Nothing) = nothing

@inline function require_context(rec::RuntimeRecords.AbstractStepRecord)
    hasproperty(rec, :context) || throw(ArgumentError(
        "Learner requires a runtime record with a `context` field, got $(typeof(rec))."
    ))
    return getproperty(rec, :context)
end

function reset!(learner::AbstractLearner, state::AbstractLearnerState)
    throw(MethodError(reset!, (learner, state)))
end

"""
Choose/sample an action.

`record` is the runtime input. For non-contextual learners it may be `nothing`.
For contextual learners it should usually be a `Context*Record`.
"""
function act!(learner::AbstractLearner,
              state::AbstractLearnerState,
              record::Union{Nothing,RuntimeRecords.AbstractStepRecord} = nothing,
              rng::AbstractRNG = Random.default_rng())
    throw(MethodError(act!, (learner, state, record, rng)))
end

"""
Update learner state from a realized runtime record.
"""
function update!(learner::AbstractLearner,
                 state::AbstractLearnerState,
                 rec::RuntimeRecords.AbstractStepRecord)
    throw(MethodError(update!, (learner, state, rec)))
end

function update_episode!(learner::AbstractLearner,
                         state::AbstractLearnerState,
                         traj::RuntimeRecords.AbstractTrajectory)
    throw(MethodError(update_episode!, (learner, state, traj)))
end

"""
Write the learner's current action distribution into `dest`.

`record` is optional runtime input; use it for contextual learners.
"""
function strategy!(dest,
                 learner::AbstractLearner,
                 state::AbstractLearnerState,
                 record::Union{Nothing,RuntimeRecords.AbstractStepRecord} = nothing)
    throw(MethodError(strategy!, (dest, learner, state, record)))
end

requires_feedback_type(::AbstractLearner) = RuntimeRecords.AbstractStepRecord
supports_action_space(::AbstractLearner) = :unknown

function Base.copy(x::Tuple{AbstractLearner,AbstractLearnerState,Union{Nothing,RuntimeRecords.AbstractStepRecord}})
    learner, state, record = x
    throw(ArgumentError(
        "No generic allocation policy helper is defined for $(typeof(learner)). Implement `strategy!`."
    ))
end

end