module Kernel

using Random

export AbstractGame, AbstractState
export AbstractFixedGame

export NodeKind, DECISION, SIMULTANEOUS, CHANCE, TERMINAL

export AbstractActionMode, IndexedActions, ExplicitActions
export action_mode

export reward_type

export SampleChance, ChanceOutcome
export JointAction, joint_action

export num_players, player_ids
export init_state, node_kind, current_player, active_players
export legal_actions, legal_action_mask, indexed_action_count
export step, observe, is_terminal

export has_action_mask

abstract type AbstractGame end
abstract type AbstractState end

abstract type AbstractFixedGame{N,R} <: AbstractGame end

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

struct JointAction{N,A<:Tuple}
    actions::A
    function JointAction{N}(actions::A) where {N,A<:Tuple}
        length(actions) == N ||
            throw(ArgumentError("Expected $N actions, got $(length(actions))."))
        new{N,A}(actions)
    end
end

JointAction(actions::A) where {A<:Tuple} = JointAction{length(actions)}(actions)
joint_action(actions::Tuple) = JointAction(actions)
joint_action(actions...) = JointAction(actions)

Base.getindex(a::JointAction{N}, i::Int) where {N} = a.actions[i]
Base.length(::JointAction{N}) where {N} = N
Base.iterate(a::JointAction, st...) = iterate(a.actions, st...)
Base.Tuple(a::JointAction) = a.actions

num_players(::AbstractFixedGame{N}) where {N} = N
player_ids(::AbstractFixedGame{N}) where {N} = Base.OneTo(N)

action_mode(::Type{<:AbstractGame}) = ExplicitActions
reward_type(::Type{<:AbstractFixedGame{N,R}}) where {N,R} = R
reward_type(::Type{<:AbstractGame}) =
    error("reward_type is not defined for this game type.")

has_action_mask(::Type{<:AbstractGame}) = false

init_state(game::AbstractGame, rng::AbstractRNG = Random.default_rng()) =
    error("init_state not implemented for $(typeof(game)).")

node_kind(game::AbstractGame, state)::NodeKind =
    error("node_kind not implemented for $(typeof(game)), $(typeof(state)).")

current_player(game::AbstractGame, state)::Int =
    error("current_player not implemented for $(typeof(game)), $(typeof(state)).")

active_players(game::AbstractGame, state) =
    error("active_players not implemented for $(typeof(game)), $(typeof(state)).")

legal_actions(game::AbstractGame, state, player::Int) =
    error("legal_actions not implemented for $(typeof(game)), player $player.")

legal_action_mask(game::AbstractGame, state, player::Int) =
    error("legal_action_mask not implemented for $(typeof(game)).")

"""
Required only for IndexedActions games.

Returns the size of the canonical indexed action domain for player `player`.
Legal indexed actions must be integers in `1:indexed_action_count(...)`.
"""
indexed_action_count(game::AbstractGame, player::Int) =
    error("indexed_action_count not implemented for indexed-action game $(typeof(game)).")

"""
Minimal hot-path transition.

Returns:
    next_state, rewards
"""
step(game::AbstractGame, state, action, rng::AbstractRNG = Random.default_rng()) =
    error("step not implemented for $(typeof(game)).")

observe(game::AbstractGame, state, player::Int) =
    error("observe not implemented for $(typeof(game)).")

is_terminal(game::AbstractGame, state) = node_kind(game, state) == TERMINAL

end