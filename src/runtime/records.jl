module RuntimeRecords

export AbstractRecord
export AbstractStepRecord
export AbstractFeedbackRecord
export AbstractTransition
export AbstractTrajectory

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

# ----------------------------------------------------------------------
# Single-agent one-step feedback records
# ----------------------------------------------------------------------

"""
Interaction record for stateless / unconditioned one-step feedback.

Best match:
- bandits
- stateless repeated play
- unconditioned strategies
"""
struct BanditRecord{A,R} <: AbstractFeedbackRecord
    action::A
    reward::R
    done::Bool
end

"""
Interaction record for semi-bandit feedback without context.

The agent chooses a possibly structured or composite action and the
environment reveals feedback for the selected components of that action,
but not full feedback for all possible actions.

Best match:
- non-contextual combinatorial bandits
- repeated slate selection without side information
- structured action problems with decomposed partial feedback
"""
struct SemiBanditRecord{A,F} <: AbstractFeedbackRecord
    action::A
    feedback::F
    done::Bool
end

"""
Interaction record for contextual bandit feedback.

The environment reveals a context before acting, the agent chooses an
action, and the environment reveals feedback only for the chosen action.

Best match:
- contextual bandits
- associative bandits
- one-step decision problems with side information
- supervised online decision problems with partial feedback

Mathematically:
- `(x_t, a_t, r_t(a_t))`

Notes:
- `context` is pre-action information used to condition the decision.
- Unlike a state transition, this record does not assert successor-state
  semantics or controlled dynamics.
"""
struct ContextBanditRecord{C,A,R} <: AbstractFeedbackRecord
    context::C
    action::A
    reward::R
    done::Bool
end

"""
Interaction record for contextual semi-bandit feedback.

The environment reveals a context before acting, the agent chooses a
possibly structured or composite action, and the environment reveals
feedback for the chosen components of that action, but not full feedback
for all possible actions.

Best match:
- combinatorial bandits
- slate / ranking feedback with per-item outcomes
- structured action problems with partial decomposed feedback

Mathematically:
- `(x_t, a_t, y_t)`
  where `y_t` is the revealed semi-bandit feedback associated with the
  selected action components

Notes:
- `feedback` is intentionally generic and may be a tuple, vector, named
  tuple, or other structured object.
- This is still a one-step feedback record and does not imply transition
  dynamics.
"""
struct ContextSemiBanditRecord{C,A,F} <: AbstractFeedbackRecord
    context::C
    action::A
    feedback::F
    done::Bool
end

"""
Interaction record for full-information feedback without context.

After the agent acts, the environment reveals the full reward/loss object
over all available actions for that round.

Best match:
- experts / Hedge style repeated decision problems
- non-contextual online learning with full feedback
- repeated play where the full payoff vector is revealed each round

Mathematically:
- `(a_t, f_t)` or equivalently just `f_t`
  where `f_t(a)` is available for all actions `a`

Notes:
- `action` is included for logging and auditing, even though it may be
  redundant once the full feedback object is revealed.
- `feedback` typically stores the full reward vector, loss vector, or
  equivalent action-indexed payoff object.
"""
struct FullInformationRecord{A,F} <: AbstractFeedbackRecord
    action::A
    feedback::F
    done::Bool
end

"""
Interaction record for contextual full-information feedback.

The environment reveals a context before acting, the agent chooses an
action, and the environment then reveals the full reward/loss object over
all available actions for that context/round.

Best match:
- contextual online learning with full feedback
- cost-sensitive classification with full label/action costs revealed
- contextual decision problems where all action outcomes are observed

Mathematically:
- `(x_t, a_t, f_t)` or equivalently `(x_t, f_t)`
  where `f_t(a)` is available for all actions `a`

Notes:
- `context` is pre-action side information, not a Markov state with
  successor semantics.
- `action` is retained for logging, diagnostics, and compatibility with
  common runtime APIs, even though it is redundant given full feedback.
"""
struct ContextFullInformationRecord{C,A,F} <: AbstractFeedbackRecord
    context::C
    action::A
    feedback::F
    done::Bool
end

# ----------------------------------------------------------------------
# Single-agent transition records
# ----------------------------------------------------------------------

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
# Multi-agent / profile one-step feedback records
# ----------------------------------------------------------------------

"""
Joint stateless one-step feedback.

Best match:
- unconditioned strategy profiles
- repeated-play / matrix-game logs
"""
struct JointBanditRecord{A,R} <: AbstractFeedbackRecord
    action::A
    reward::R
    done::Bool
end

"""
Joint semi-bandit feedback without context.

Best match:
- structured profile actions with decomposed partial feedback
- non-contextual multi-agent combinatorial feedback
"""
struct JointSemiBanditRecord{A,F} <: AbstractFeedbackRecord
    action::A
    feedback::F
    done::Bool
end

"""
Joint contextual bandit interaction.

Best match:
- contextual repeated games with partial feedback
- multi-agent one-step decision problems with side information
- profile-based interaction where only realized payoffs are revealed
"""
struct JointContextBanditRecord{C,A,R} <: AbstractFeedbackRecord
    context::C
    action::A
    reward::R
    done::Bool
end

"""
Joint contextual semi-bandit interaction.

Best match:
- structured multi-agent action profiles with decomposed partial feedback
- slate/profile settings where only selected components reveal outcomes
"""
struct JointContextSemiBanditRecord{C,A,F} <: AbstractFeedbackRecord
    context::C
    action::A
    feedback::F
    done::Bool
end

"""
Joint full-information interaction without context.

Best match:
- repeated normal-form games with full payoff feedback
- profile-based online learning with full round feedback
"""
struct JointFullInformationRecord{A,F} <: AbstractFeedbackRecord
    action::A
    feedback::F
    done::Bool
end

"""
Joint contextual full-information interaction.

Best match:
- contextual repeated games with full payoff feedback
- multi-agent online learning where all profile/action outcomes are
  revealed after each round
"""
struct JointContextFullInformationRecord{C,A,F} <: AbstractFeedbackRecord
    context::C
    action::A
    feedback::F
    done::Bool
end

# ----------------------------------------------------------------------
# Multi-agent / profile transition records
# ----------------------------------------------------------------------

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

function contexts(tr)
    return [_getprop(r, :context) for r in tr if _hasprop(r, :context)]
end

function feedbacks(tr)
    return [_getprop(r, :feedback) for r in tr if _hasprop(r, :feedback)]
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