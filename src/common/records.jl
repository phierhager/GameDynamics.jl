module RuntimeRecords

export AbstractRecord
export AbstractStepRecord
export AbstractFeedbackRecord
export AbstractTransition
export AbstractTrajectory

export AbstractSingleAgentRecord
export AbstractJointRecord
export AbstractSingleAgentFeedbackRecord
export AbstractJointFeedbackRecord
export AbstractSingleAgentTransitionRecord
export AbstractJointTransitionRecord

export AbstractAgentMode
export SingleAgentMode
export JointAgentMode

export AbstractFeedbackMode
export BanditFeedback
export SemiBanditFeedback
export FullInformationFeedback
export TransitionFeedback

export AbstractContextMode
export NoContextMode
export ContextMode

export AbstractStateMode
export NoStateMode
export StateOnlyMode
export ObservationOnlyMode
export StateObservationMode

export agent_mode
export feedback_mode
export context_mode
export state_mode

export BanditRecord
export SemiBanditRecord
export ContextBanditRecord
export ContextSemiBanditRecord
export FullInformationRecord
export ContextFullInformationRecord

export StateTransition
export ObservationTransition
export StateObservationTransition

export JointBanditRecord
export JointSemiBanditRecord
export JointContextBanditRecord
export JointContextSemiBanditRecord
export JointFullInformationRecord
export JointContextFullInformationRecord

export JointStateTransition
export JointObservationTransition
export JointStateObservationTransition

export EpisodeTrajectory

export push!
export reset!
export length
export isempty

export rewards
export actions
export states
export observations
export next_states
export next_observations
export contexts
export feedbacks

export discounted_return
export undiscounted_return
export player_discounted_return
export player_undiscounted_return

# ----------------------------------------------------------------------
# Abstract roots
# ----------------------------------------------------------------------

abstract type AbstractRecord end
abstract type AbstractStepRecord <: AbstractRecord end
abstract type AbstractFeedbackRecord <: AbstractStepRecord end
abstract type AbstractTransition <: AbstractStepRecord end
abstract type AbstractTrajectory <: AbstractRecord end

abstract type AbstractSingleAgentRecord <: AbstractStepRecord end
abstract type AbstractJointRecord <: AbstractStepRecord end

abstract type AbstractSingleAgentFeedbackRecord <: AbstractFeedbackRecord end
abstract type AbstractJointFeedbackRecord <: AbstractFeedbackRecord end

abstract type AbstractSingleAgentTransitionRecord <: AbstractTransition end
abstract type AbstractJointTransitionRecord <: AbstractTransition end

# ----------------------------------------------------------------------
# Traits
# ----------------------------------------------------------------------

abstract type AbstractAgentMode end
struct SingleAgentMode <: AbstractAgentMode end
struct JointAgentMode <: AbstractAgentMode end

abstract type AbstractFeedbackMode end
struct BanditFeedback <: AbstractFeedbackMode end
struct SemiBanditFeedback <: AbstractFeedbackMode end
struct FullInformationFeedback <: AbstractFeedbackMode end
struct TransitionFeedback <: AbstractFeedbackMode end

abstract type AbstractContextMode end
struct NoContextMode <: AbstractContextMode end
struct ContextMode <: AbstractContextMode end

abstract type AbstractStateMode end
struct NoStateMode <: AbstractStateMode end
struct StateOnlyMode <: AbstractStateMode end
struct ObservationOnlyMode <: AbstractStateMode end
struct StateObservationMode <: AbstractStateMode end

agent_mode(::Type{<:AbstractSingleAgentRecord}) = SingleAgentMode()
agent_mode(::Type{<:AbstractJointRecord}) = JointAgentMode()
agent_mode(x) = agent_mode(typeof(x))

feedback_mode(::Type{<:AbstractTransition}) = TransitionFeedback()
feedback_mode(x) = feedback_mode(typeof(x))

context_mode(::Type{<:AbstractStepRecord}) = NoContextMode()
context_mode(x) = context_mode(typeof(x))

state_mode(::Type{<:AbstractFeedbackRecord}) = NoStateMode()
state_mode(x) = state_mode(typeof(x))

# ----------------------------------------------------------------------
# Single-agent one-step feedback records
# ----------------------------------------------------------------------

struct BanditRecord{A,R} <: AbstractSingleAgentFeedbackRecord
    action::A
    reward::R
    done::Bool
end

struct SemiBanditRecord{A,F} <: AbstractSingleAgentFeedbackRecord
    action::A
    feedback::F
    done::Bool
end

struct ContextBanditRecord{C,A,R} <: AbstractSingleAgentFeedbackRecord
    context::C
    action::A
    reward::R
    done::Bool
end

struct ContextSemiBanditRecord{C,A,F} <: AbstractSingleAgentFeedbackRecord
    context::C
    action::A
    feedback::F
    done::Bool
end

struct FullInformationRecord{A,F} <: AbstractSingleAgentFeedbackRecord
    action::A
    feedback::F
    done::Bool
end

struct ContextFullInformationRecord{C,A,F} <: AbstractSingleAgentFeedbackRecord
    context::C
    action::A
    feedback::F
    done::Bool
end

feedback_mode(::Type{<:BanditRecord}) = BanditFeedback()
feedback_mode(::Type{<:ContextBanditRecord}) = BanditFeedback()

feedback_mode(::Type{<:SemiBanditRecord}) = SemiBanditFeedback()
feedback_mode(::Type{<:ContextSemiBanditRecord}) = SemiBanditFeedback()

feedback_mode(::Type{<:FullInformationRecord}) = FullInformationFeedback()
feedback_mode(::Type{<:ContextFullInformationRecord}) = FullInformationFeedback()

context_mode(::Type{<:ContextBanditRecord}) = ContextMode()
context_mode(::Type{<:ContextSemiBanditRecord}) = ContextMode()
context_mode(::Type{<:ContextFullInformationRecord}) = ContextMode()

# ----------------------------------------------------------------------
# Single-agent transition records
# ----------------------------------------------------------------------

struct StateTransition{S,A,R} <: AbstractSingleAgentTransitionRecord
    state::S
    action::A
    reward::R
    next_state::S
    done::Bool
end

struct ObservationTransition{O,A,R} <: AbstractSingleAgentTransitionRecord
    observation::O
    action::A
    reward::R
    next_observation::O
    done::Bool
end

struct StateObservationTransition{S,O,A,R} <: AbstractSingleAgentTransitionRecord
    state::S
    observation::O
    action::A
    reward::R
    next_state::S
    next_observation::O
    done::Bool
end

state_mode(::Type{<:StateTransition}) = StateOnlyMode()
state_mode(::Type{<:ObservationTransition}) = ObservationOnlyMode()
state_mode(::Type{<:StateObservationTransition}) = StateObservationMode()

# ----------------------------------------------------------------------
# Multi-agent one-step feedback records
# ----------------------------------------------------------------------

struct JointBanditRecord{A,R} <: AbstractJointFeedbackRecord
    action::A
    reward::R
    done::Bool
end

struct JointSemiBanditRecord{A,F} <: AbstractJointFeedbackRecord
    action::A
    feedback::F
    done::Bool
end

struct JointContextBanditRecord{C,A,R} <: AbstractJointFeedbackRecord
    context::C
    action::A
    reward::R
    done::Bool
end

struct JointContextSemiBanditRecord{C,A,F} <: AbstractJointFeedbackRecord
    context::C
    action::A
    feedback::F
    done::Bool
end

struct JointFullInformationRecord{A,F} <: AbstractJointFeedbackRecord
    action::A
    feedback::F
    done::Bool
end

struct JointContextFullInformationRecord{C,A,F} <: AbstractJointFeedbackRecord
    context::C
    action::A
    feedback::F
    done::Bool
end

feedback_mode(::Type{<:JointBanditRecord}) = BanditFeedback()
feedback_mode(::Type{<:JointContextBanditRecord}) = BanditFeedback()

feedback_mode(::Type{<:JointSemiBanditRecord}) = SemiBanditFeedback()
feedback_mode(::Type{<:JointContextSemiBanditRecord}) = SemiBanditFeedback()

feedback_mode(::Type{<:JointFullInformationRecord}) = FullInformationFeedback()
feedback_mode(::Type{<:JointContextFullInformationRecord}) = FullInformationFeedback()

context_mode(::Type{<:JointContextBanditRecord}) = ContextMode()
context_mode(::Type{<:JointContextSemiBanditRecord}) = ContextMode()
context_mode(::Type{<:JointContextFullInformationRecord}) = ContextMode()

# ----------------------------------------------------------------------
# Multi-agent transition records
# ----------------------------------------------------------------------

struct JointStateTransition{S,A,R} <: AbstractJointTransitionRecord
    state::S
    action::A
    reward::R
    next_state::S
    done::Bool
end

struct JointObservationTransition{O,A,R} <: AbstractJointTransitionRecord
    observation::O
    action::A
    reward::R
    next_observation::O
    done::Bool
end

struct JointStateObservationTransition{S,O,A,R} <: AbstractJointTransitionRecord
    state::S
    observation::O
    action::A
    reward::R
    next_state::S
    next_observation::O
    done::Bool
end

state_mode(::Type{<:JointStateTransition}) = StateOnlyMode()
state_mode(::Type{<:JointObservationTransition}) = ObservationOnlyMode()
state_mode(::Type{<:JointStateObservationTransition}) = StateObservationMode()

# ----------------------------------------------------------------------
# Trajectory container
# ----------------------------------------------------------------------

mutable struct EpisodeTrajectory{T<:AbstractStepRecord,V<:AbstractVector{T}} <: AbstractTrajectory
    records::V
    terminated::Bool
    truncated::Bool
end

function EpisodeTrajectory(::Type{T}) where {T<:AbstractStepRecord}
    return EpisodeTrajectory{T,Vector{T}}(T[], false, false)
end

Base.length(tr::EpisodeTrajectory) = length(tr.records)
Base.isempty(tr::EpisodeTrajectory) = isempty(tr.records)
Base.getindex(tr::EpisodeTrajectory, i::Int) = tr.records[i]
Base.iterate(tr::EpisodeTrajectory, st...) = iterate(tr.records, st...)

function Base.push!(tr::EpisodeTrajectory{T}, rec::T) where {T<:AbstractStepRecord}
    push!(tr.records, rec)
    return tr
end

function reset!(tr::EpisodeTrajectory)
    empty!(tr.records)
    tr.terminated = false
    tr.truncated = false
    return tr
end

# ----------------------------------------------------------------------
# Field access
# ----------------------------------------------------------------------

@inline _hasprop(x, s::Symbol) = hasproperty(x, s)
@inline _getprop(x, s::Symbol) = getproperty(x, s)

rewards(tr) = [_getprop(r, :reward) for r in tr if _hasprop(r, :reward)]
actions(tr) = [_getprop(r, :action) for r in tr if _hasprop(r, :action)]
states(tr) = [_getprop(r, :state) for r in tr if _hasprop(r, :state)]
observations(tr) = [_getprop(r, :observation) for r in tr if _hasprop(r, :observation)]
next_states(tr) = [_getprop(r, :next_state) for r in tr if _hasprop(r, :next_state)]
next_observations(tr) = [_getprop(r, :next_observation) for r in tr if _hasprop(r, :next_observation)]
contexts(tr) = [_getprop(r, :context) for r in tr if _hasprop(r, :context)]
feedbacks(tr) = [_getprop(r, :feedback) for r in tr if _hasprop(r, :feedback)]

# ----------------------------------------------------------------------
# Return helpers
# ----------------------------------------------------------------------

@inline function _reward_component(r::Real, player::Int)
    player == 1 || throw(ArgumentError("Scalar reward only supports player 1."))
    return Float64(r)
end

@inline _reward_component(r::Tuple, player::Int) = Float64(r[player])
@inline _reward_component(r::AbstractVector, player::Int) = Float64(r[player])

function undiscounted_return(tr)
    acc = 0.0
    for rec in tr
        _hasprop(rec, :reward) || continue
        r = _getprop(rec, :reward)
        r isa Real || throw(ArgumentError(
            "Trajectory contains non-scalar rewards; use player-specific return helpers."
        ))
        acc += Float64(r)
    end
    return acc
end

function discounted_return(tr; discount::Float64 = 1.0)
    0.0 <= discount <= 1.0 || throw(ArgumentError("discount must be in [0,1]."))
    acc = 0.0
    coeff = 1.0
    for rec in tr
        _hasprop(rec, :reward) || continue
        r = _getprop(rec, :reward)
        r isa Real || throw(ArgumentError(
            "Trajectory contains non-scalar rewards; use player-specific return helpers."
        ))
        acc += coeff * Float64(r)
        coeff *= discount
    end
    return acc
end

function player_undiscounted_return(tr, player::Int)
    acc = 0.0
    for rec in tr
        _hasprop(rec, :reward) || continue
        acc += _reward_component(_getprop(rec, :reward), player)
    end
    return acc
end

function player_discounted_return(tr, player::Int; discount::Float64 = 1.0)
    0.0 <= discount <= 1.0 || throw(ArgumentError("discount must be in [0,1]."))
    acc = 0.0
    coeff = 1.0
    for rec in tr
        _hasprop(rec, :reward) || continue
        acc += coeff * _reward_component(_getprop(rec, :reward), player)
        coeff *= discount
    end
    return acc
end

end