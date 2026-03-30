module RuntimeTrajectories

export AbstractTrajectoryRecord
export AbstractTransition
export AbstractTrajectory

export BanditTransition
export StateTransition
export ObservationTransition
export StateObservationTransition

export JointBanditTransition
export JointStateTransition
export JointObservationTransition
export JointStateObservationTransition

export EpisodeTrajectory
export ReplayBuffer

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

export discounted_return
export undiscounted_return
export player_discounted_return
export player_undiscounted_return

# ----------------------------------------------------------------------
# Abstract roots
# ----------------------------------------------------------------------

abstract type AbstractTrajectoryRecord end
abstract type AbstractTransition <: AbstractTrajectoryRecord end
abstract type AbstractTrajectory end

# ----------------------------------------------------------------------
# Single-agent trajectory records
# ----------------------------------------------------------------------

"""
Interaction record for stateless / unconditioned control.

Best match:
- bandits
- stateless repeated play
- unconditioned strategies
"""
struct BanditTransition{A,R} <: AbstractTransition
    action::A
    reward::R
    done::Bool
end

"""
Transition for state-conditioned interaction.

Best match:
- MDP-style control
- state-conditioned strategies
"""
struct StateTransition{S,A,R} <: AbstractTransition
    state::S
    action::A
    reward::R
    next_state::S
    done::Bool
end

"""
Transition for observation-conditioned / partial-observation control.

Best match:
- POMDP-style control
- observation-conditioned strategies
"""
struct ObservationTransition{O,A,R} <: AbstractTransition
    observation::O
    action::A
    reward::R
    next_observation::O
    done::Bool
end

"""
Transition carrying both latent state and observation.

Best match:
- training with simulator state access while the acting strategy uses observations
- actor-critic / model-based diagnostics in partially observed settings
"""
struct StateObservationTransition{S,O,A,R} <: AbstractTransition
    state::S
    observation::O
    action::A
    reward::R
    next_state::S
    next_observation::O
    done::Bool
end

# ----------------------------------------------------------------------
# Multi-agent / profile trajectory records
# ----------------------------------------------------------------------

"""
Joint stateless interaction.

Best match:
- unconditioned strategy profiles
- repeated-play / matrix-game logs
"""
struct JointBanditTransition{A,R} <: AbstractTransition
    action::A
    reward::R
    done::Bool
end

"""
Joint transition with full state.

Best match:
- state-conditioned strategy profiles
- Markov games with state-based centralized training/control
"""
struct JointStateTransition{S,A,R} <: AbstractTransition
    state::S
    action::A
    reward::R
    next_state::S
    done::Bool
end

"""
Joint transition with local/joint observations.

Best match:
- observation-conditioned strategy profiles
- POSG / Dec-POMDP style decentralized execution
"""
struct JointObservationTransition{O,A,R} <: AbstractTransition
    observation::O
    action::A
    reward::R
    next_observation::O
    done::Bool
end

"""
Joint transition carrying both state and local/joint observations.

Best match:
- centralized-training decentralized-execution pipelines
- partially observed multi-agent training with simulator state access
"""
struct JointStateObservationTransition{S,O,A,R} <: AbstractTransition
    state::S
    observation::O
    action::A
    reward::R
    next_state::S
    next_observation::O
    done::Bool
end

# ----------------------------------------------------------------------
# Trajectory containers
# ----------------------------------------------------------------------

mutable struct EpisodeTrajectory{T<:AbstractTrajectoryRecord,V<:AbstractVector{T}} <: AbstractTrajectory
    records::V
    terminated::Bool
    truncated::Bool
end

function EpisodeTrajectory(::Type{T}) where {T<:AbstractTrajectoryRecord}
    return EpisodeTrajectory{T,Vector{T}}(T[], false, false)
end

Base.length(tr::EpisodeTrajectory) = length(tr.records)
Base.isempty(tr::EpisodeTrajectory) = isempty(tr.records)
Base.getindex(tr::EpisodeTrajectory, i::Int) = tr.records[i]
Base.iterate(tr::EpisodeTrajectory, st...) = iterate(tr.records, st...)

function Base.push!(tr::EpisodeTrajectory{T}, rec::T) where {T<:AbstractTrajectoryRecord}
    push!(tr.records, rec)
    return tr
end

function reset!(tr::EpisodeTrajectory)
    empty!(tr.records)
    tr.terminated = false
    tr.truncated = false
    return tr
end

"""
Simple append-only replay buffer.
"""
mutable struct ReplayBuffer{T<:AbstractTrajectoryRecord,V<:AbstractVector{T}}
    data::V
    capacity::Int
    position::Int
    full::Bool
end

function ReplayBuffer(::Type{T}, capacity::Int) where {T<:AbstractTrajectoryRecord}
    capacity > 0 || throw(ArgumentError("capacity must be positive."))
    return ReplayBuffer{T,Vector{T}}(Vector{T}(undef, capacity), capacity, 1, false)
end

Base.length(buf::ReplayBuffer) = buf.full ? buf.capacity : (buf.position - 1)
Base.isempty(buf::ReplayBuffer) = length(buf) == 0

function Base.push!(buf::ReplayBuffer{T}, rec::T) where {T<:AbstractTrajectoryRecord}
    buf.data[buf.position] = rec
    if buf.position == buf.capacity
        buf.position = 1
        buf.full = true
    else
        buf.position += 1
    end
    return buf
end

function reset!(buf::ReplayBuffer)
    buf.position = 1
    buf.full = false
    return buf
end

# ----------------------------------------------------------------------
# Field access helpers
# ----------------------------------------------------------------------

@inline _hasprop(x, s::Symbol) = hasproperty(x, s)
@inline _getprop(x, s::Symbol) = getproperty(x, s)

function rewards(tr)
    return [_getprop(r, :reward) for r in tr if _hasprop(r, :reward)]
end

function actions(tr)
    return [_getprop(r, :action) for r in tr if _hasprop(r, :action)]
end

function states(tr)
    return [_getprop(r, :state) for r in tr if _hasprop(r, :state)]
end

function observations(tr)
    return [_getprop(r, :observation) for r in tr if _hasprop(r, :observation)]
end

function next_states(tr)
    return [_getprop(r, :next_state) for r in tr if _hasprop(r, :next_state)]
end

function next_observations(tr)
    return [_getprop(r, :next_observation) for r in tr if _hasprop(r, :next_observation)]
end

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
        r isa Real || throw(ArgumentError("Trajectory contains non-scalar rewards; use player-specific return helpers."))
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
        r isa Real || throw(ArgumentError("Trajectory contains non-scalar rewards; use player-specific return helpers."))
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