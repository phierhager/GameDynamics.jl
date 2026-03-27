module Kernel

using Random

export AbstractGame, AbstractState

export NodeKind, DECISION, SIMULTANEOUS, CHANCE, TERMINAL

export AbstractActionMode, IndexedActions, ExplicitActions
export action_mode
export reward_type

export SampleChance, ChanceOutcome

export JointAction, joint_action
export action_for_player, validate_joint_action

export num_players, player_ids
export init_state, node_kind
export legal_actions, legal_action_mask, indexed_action_count
export step, observe, is_terminal
export has_action_mask
export current_player, active_players, acting_players, only_acting_player

abstract type AbstractGame{N,R} end
abstract type AbstractState end

@enum NodeKind::UInt8 begin
    DECISION     = 0x01
    SIMULTANEOUS = 0x02
    CHANCE       = 0x03
    TERMINAL     = 0x04
end

abstract type AbstractActionMode end
struct IndexedActions <: AbstractActionMode end
struct ExplicitActions <: AbstractActionMode end

struct SampleChance end
struct ChanceOutcome{E}
    event::E
end

"""
Partial simultaneous action aligned positionally with `active_players(game, state)`.

Semantics:
- `ja[i]` means the action for the `i`-th active player
- it stores only actions, not player ids
- active-player order is defined by `active_players(game, state)`

This is the single kernel simultaneous-action primitive.
"""
struct JointAction{A<:Tuple}
    actions::A
end

joint_action(actions::Tuple) = JointAction(actions)
joint_action(actions::AbstractVector) = JointAction(Tuple(actions))
joint_action(actions...) = JointAction(actions)

Base.getindex(a::JointAction, i::Int) = a.actions[i]
Base.length(a::JointAction) = length(a.actions)
Base.iterate(a::JointAction, st...) = iterate(a.actions, st...)
Base.Tuple(a::JointAction) = a.actions
Base.firstindex(::JointAction) = 1
Base.lastindex(a::JointAction) = length(a)

num_players(::AbstractGame{N}) where {N} = N
reward_type(::Type{<:AbstractGame{N,R}}) where {N,R} = R
player_ids(::AbstractGame{N}) where {N} = Base.OneTo(N)

action_mode(::Type{<:AbstractGame}) = ExplicitActions
has_action_mask(::Type{<:AbstractGame}) = false

init_state(game::AbstractGame, rng::AbstractRNG = Random.default_rng()) =
    error("init_state not implemented for $(typeof(game)).")

node_kind(game::AbstractGame, state)::NodeKind =
    error("node_kind not implemented for $(typeof(game)), $(typeof(state)).")

"""
Return the acting player at a `DECISION` node.

Game implementations should define this only for decision nodes.
Consumer code that wants a node-kind-agnostic answer should usually call
`acting_players(game, state)` instead.
"""
current_player(game::AbstractGame, state)::Int =
    throw(ArgumentError(
        "current_player is only defined for DECISION nodes; got $(node_kind(game, state)) for $(typeof(game))."
    ))

"""
Return the acting players at a `SIMULTANEOUS` node.

Required invariants:
- valid player ids
- strictly ascending order
- one slot per active player

Consumer code that wants a node-kind-agnostic answer should usually call
`acting_players(game, state)` instead.
"""
active_players(game::AbstractGame, state) =
    throw(ArgumentError(
        "active_players is only defined for SIMULTANEOUS nodes; got $(node_kind(game, state)) for $(typeof(game))."
    ))

legal_actions(game::AbstractGame, state, player::Int) =
    error("legal_actions not implemented for $(typeof(game)), player $player.")

legal_action_mask(game::AbstractGame, state, player::Int) =
    error("legal_action_mask not implemented for $(typeof(game)).")

indexed_action_count(game::AbstractGame, player::Int) =
    error("indexed_action_count not implemented for indexed-action game $(typeof(game)).")

"""
Minimal hot-path transition.

Simultaneous nodes must accept the kernel partial `JointAction` whose entries align
with `active_players(game, state)`.
"""
step(game::AbstractGame, state, action, rng::AbstractRNG = Random.default_rng()) =
    error("step not implemented for $(typeof(game)).")

observe(game::AbstractGame, state, player::Int) =
    error("observe not implemented for $(typeof(game)).")

is_terminal(game::AbstractGame, state) = node_kind(game, state) == TERMINAL

@inline function _local_active_player_index(aps, p::Int)
    @inbounds for i in eachindex(aps)
        aps[i] == p && return i
    end
    return 0
end

"""
Return the action assigned to player `p` in a simultaneous `JointAction`,
or `nothing` when `p` is inactive.
"""
function action_for_player(game::AbstractGame, state, ja::JointAction, p::Int)
    aps = active_players(game, state)
    idx = _local_active_player_index(aps, p)
    return idx == 0 ? nothing : ja[idx]
end

@inline function _validate_active_players_order(aps)
    @inbounds for i in 2:length(aps)
        aps[i - 1] < aps[i] || throw(ArgumentError(
            "active_players(game, state) must return players in strictly ascending player-id order; got $(Tuple(aps))."
        ))
    end
    return aps
end

"""
Return the players who are expected to act at the current node.

Semantics:
- at `DECISION` nodes, this is `(current_player(game, state),)`
- at `SIMULTANEOUS` nodes, this is `active_players(game, state)`
- otherwise, this is `()`

This is the general public helper that consumer code should usually call.
Game implementations should still specialize `current_player` and
`active_players` as the node-specific kernel primitives.
"""
@inline function acting_players(game::AbstractGame, state)
    nk = node_kind(game, state)
    if nk == DECISION
        return (current_player(game, state),)
    elseif nk == SIMULTANEOUS
        return active_players(game, state)
    else
        return ()
    end
end

@inline function only_acting_player(game::AbstractGame, state)
    aps = acting_players(game, state)
    length(aps) == 1 || throw(ArgumentError(
        "Expected exactly one acting player, got $(Tuple(aps))."
    ))
    return only(aps)
end

"""
Canonical joint-action validation path.

A valid simultaneous `JointAction` must contain exactly one action for each active
player, in the same order as `active_players(game, state)`.
"""
function validate_joint_action(game::AbstractGame, state, ja::JointAction)
    nk = node_kind(game, state)
    nk == SIMULTANEOUS || throw(ArgumentError(
        "validate_joint_action is only valid at simultaneous nodes; got $(nk)."
    ))

    aps = _validate_active_players_order(active_players(game, state))
    length(ja) == length(aps) || throw(ArgumentError(
        "JointAction arity mismatch: expected $(length(aps)) actions for active_players=$(Tuple(aps)), got $(length(ja))."
    ))

    @inbounds for i in eachindex(aps)
        p = aps[i]
        a = ja[i]
        legal = legal_actions(game, state, p)
        a in legal || throw(ArgumentError(
            "Illegal action $a for active player $p at local slot $i."
        ))
    end

    return ja
end

end