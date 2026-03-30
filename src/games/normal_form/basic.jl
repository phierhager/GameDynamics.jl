module NormalForm

using Random
using ..Kernel
using ..RuntimeRecords
using ..Enumerative
using ..Families

export NormalFormState
export NormalFormGame

export payoff_tensor
export action_count
export action_counts
export pure_payoff
export support_profiles

"""
Single-stage state for a normal-form game.

Semantics:
- `played == false`: simultaneous move still pending
- `played == true`: terminal
"""
struct NormalFormState <: Kernel.AbstractState
    played::Bool
end

"""
Generic finite normal-form game.

Fields:
- `payoffs[p]`: player-`p` payoff tensor
- `action_sizes[p]`: number of actions for player `p`

Conventions:
- one simultaneous player-controlled move
- then terminal
- rewards are returned as `NTuple{N,Float64}`
"""
struct NormalFormGame{N,T<:Tuple,R} <: Kernel.AbstractGame{N,R}
    payoffs::T
    action_sizes::NTuple{N,Int}
end

# ----------------------------------------------------------------------
# Constructors
# ----------------------------------------------------------------------

function NormalFormGame(payoffs::T) where {T<:Tuple}
    N = length(payoffs)
    N > 0 || throw(ArgumentError("NormalFormGame requires at least one player."))

    dims = size(payoffs[1])
    length(dims) == N || throw(ArgumentError(
        "Each payoff tensor must have one axis per player; got $(length(dims)) axes for $N players."
    ))

    @inbounds for p in 2:N
        size(payoffs[p]) == dims || throw(ArgumentError(
            "All payoff tensors must have identical shape."
        ))
    end

    @inbounds for i in 1:N
        dims[i] > 0 || throw(ArgumentError(
            "Each player must have at least one action."
        ))
    end

    R = NTuple{N,Float64}
    return NormalFormGame{N,T,R}(payoffs, ntuple(i -> dims[i], N))
end

# ----------------------------------------------------------------------
# Basic accessors
# ----------------------------------------------------------------------

payoff_tensor(g::NormalFormGame, player::Int) = g.payoffs[player]
action_count(g::NormalFormGame, player::Int) = g.action_sizes[player]
action_counts(g::NormalFormGame{N}) where {N} = g.action_sizes

function support_profiles(g::NormalFormGame{N}) where {N}
    ranges = ntuple(i -> Base.OneTo(g.action_sizes[i]), N)
    return Iterators.product(ranges...)
end

function pure_payoff(g::NormalFormGame{N}, profile::NTuple{N,Int}) where {N}
    @inbounds for p in 1:N
        1 <= profile[p] <= g.action_sizes[p] || throw(ArgumentError(
            "Illegal action $(profile[p]) for player $p; expected an integer in 1:$(g.action_sizes[p])."
        ))
    end
    return ntuple(p -> Float64(g.payoffs[p][profile...]), N)
end

# ----------------------------------------------------------------------
# Family / kernel declarations
# ----------------------------------------------------------------------

Families.game_family(::Type{<:NormalFormGame}) = Families.NormalFormFamily()

Kernel.record_type(::Type{<:NormalFormGame}) = RuntimeRecords.AbstractJointFeedbackRecord

Kernel.action_mode(::Type{<:NormalFormGame}) = Kernel.IndexedActions()
Kernel.has_action_mask(::Type{<:NormalFormGame}) = true

Kernel.init_state(::NormalFormGame, rng::AbstractRNG = Random.default_rng()) =
    NormalFormState(false)

Kernel.node_kind(::NormalFormGame, s::NormalFormState) =
    s.played ? Kernel.TERMINAL : Kernel.SIMULTANEOUS

Kernel.active_players(::NormalFormGame{N}, s::NormalFormState) where {N} = Base.OneTo(N)

Kernel.legal_actions(g::NormalFormGame, s::NormalFormState, player::Int) =
    Base.OneTo(g.action_sizes[player])

Kernel.indexed_action_count(g::NormalFormGame, player::Int) = g.action_sizes[player]

Kernel.legal_action_mask(g::NormalFormGame, s::NormalFormState, player::Int) =
    ntuple(_ -> true, g.action_sizes[player])

Kernel.observe(::NormalFormGame, s::NormalFormState, player::Int) = nothing

# ----------------------------------------------------------------------
# Step + record
# ----------------------------------------------------------------------

function Kernel.step(g::NormalFormGame{N},
                     s::NormalFormState,
                     a::Kernel.JointAction,
                     rng::AbstractRNG = Random.default_rng()) where {N}
    s.played && throw(ArgumentError("Cannot step from a terminal normal-form state."))

    Kernel.validate_joint_action(g, s, a)

    profile = Tuple(a)
    length(profile) == N || throw(ArgumentError(
        "NormalFormGame expected $N simultaneous actions, got $(length(profile))."
    ))

    reward = ntuple(p -> Float64(g.payoffs[p][profile...]), N)
    return NormalFormState(true), reward
end

function Kernel.make_record(g::NormalFormGame{N},
                            state::NormalFormState,
                            action::Kernel.JointAction,
                            next_state::NormalFormState,
                            reward;
                            done::Bool = Kernel.is_terminal(g, next_state)) where {N}
    return RuntimeRecords.JointBanditRecord(Tuple(action), reward, done)
end

Kernel.record_type(::Type{<:NormalFormGame{N}}) where {N} =
    RuntimeRecords.JointBanditRecord{NTuple{N,Int},NTuple{N,Float64}}

# ----------------------------------------------------------------------
# Enumerative interface
# ----------------------------------------------------------------------

function Enumerative.transition_kernel(g::NormalFormGame{N},
                                       s::NormalFormState,
                                       a::Kernel.JointAction) where {N}
    ns, r = Kernel.step(g, s, a)
    return ((ns, 1.0, r),)
end

Enumerative.terminal_payoffs(::NormalFormGame{N}, s::NormalFormState) where {N} =
    throw(ArgumentError("terminal_payoffs is only defined on terminal states reached after play."))

end