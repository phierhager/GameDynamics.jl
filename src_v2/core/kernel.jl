module Kernel

using Random

export AbstractGame, AbstractState
export AbstractFixedGame
export NodeKind, DECISION, SIMULTANEOUS, CHANCE, TERMINAL

export AbstractActionMode, IndexedActions, ExplicitActions, SpaceActions
export action_mode

export reward_type

export SampleChance, ChanceOutcome

export JointAction, joint_action

export num_players, player_ids
export init_state, node_kind, current_player, active_players
export legal_actions, legal_action_mask, num_base_actions, encode_action, decode_action
export step, step_with_info, observe, is_terminal

export validate_legal_action_mask, validate_indexed_action_contract
export default_info

abstract type AbstractGame end
abstract type AbstractState end

"""
Fixed-player hot-path game.

Parameters:
- N: number of players
- R: reward container type returned by `step`
"""
abstract type AbstractFixedGame{N,R} <: AbstractGame end

@enum NodeKind::UInt8 begin
    DECISION = 0x01
    SIMULTANEOUS = 0x02
    CHANCE = 0x03
    TERMINAL = 0x04
end

abstract type AbstractActionMode end
struct IndexedActions <: AbstractActionMode end
struct ExplicitActions <: AbstractActionMode end
struct SpaceActions <: AbstractActionMode end

"""
At chance nodes, either request internal sampling or provide an explicit outcome.
"""
struct SampleChance end

struct ChanceOutcome{E}
    event::E
end

"""
Canonical fixed-player simultaneous action representation.

Actions are ordered by `player_ids(game)`, i.e. 1, 2, ..., N.
Generic fixed-player simultaneous tooling should target this representation.
"""
struct JointAction{N,A<:Tuple}
    actions::A
    function JointAction{N}(actions::A) where {N,A<:Tuple}
        length(actions) == N || throw(ArgumentError("Expected $N actions, got $(length(actions))."))
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
Base.@propagate_inbounds

num_players(::AbstractFixedGame{N}) where {N} = N
player_ids(::AbstractFixedGame{N}) where {N} = Base.OneTo(N)

"""
Action representation mode.

Returns the action-mode type, not an instance.
"""
action_mode(::Type{<:AbstractGame}) = ExplicitActions

"""
Reward container type for the game.

For fixed-player games, the intended forms are:
- single-player: a scalar real
- multi-player: `NTuple{N,T}` or another small fixed-size concrete container
  such as `SVector{N,T}`

For generic tooling, this method is the source of truth.
"""
reward_type(::Type{<:AbstractFixedGame{N,R}}) where {N,R} = R
reward_type(::Type{<:AbstractGame}) =
    error("reward_type is not defined for this game type. Concrete game types participating in generic tooling must commit to a reward type.")

init_state(game::AbstractGame, rng::AbstractRNG = Random.default_rng()) =
    error("init_state not implemented for $(typeof(game)).")

node_kind(game::AbstractGame, state)::NodeKind =
    error("node_kind not implemented for $(typeof(game)), $(typeof(state)).")

current_player(game::AbstractGame, state)::Int =
    error("current_player not implemented for $(typeof(game)), $(typeof(state)).")

"""
For simultaneous nodes in fixed-player games, prefer:
- `Base.OneTo(N)` if all players act
- `NTuple{K,Int}` for strict subsets

Avoid vectors in hot paths.
"""
active_players(game::AbstractGame, state) =
    error("active_players not implemented for $(typeof(game)), $(typeof(state)).")

"""
Legal actions contract by mode:

- IndexedActions:
    returns a compact integer index set, preferably `Base.OneTo(K)` or an `AbstractUnitRange`
- ExplicitActions:
    returns an iterable of concrete actions
- SpaceActions:
    returns the current structured action domain directly

`legal_actions` is mandatory for all games.
"""
legal_actions(game::AbstractGame, state, player::Int) =
    error("legal_actions not implemented for $(typeof(game)), player $player.")

"""
Only meaningful for IndexedActions.

Contract:
- must be an indexable boolean-like collection
- `length(mask) == num_base_actions(game, player)`
- `mask[i] == true` iff indexed action `i` is legal
"""
legal_action_mask(game::AbstractGame, state, player::Int) =
    error("legal_action_mask not implemented for indexed-action game $(typeof(game)).")

"""
Size of the canonical indexed action domain for player `player`.
Required for IndexedActions.
"""
num_base_actions(game::AbstractGame, player::Int) =
    error("num_base_actions not implemented for indexed-action game $(typeof(game)).")

encode_action(game::AbstractGame, player::Int, action) =
    error("encode_action not implemented for $(typeof(game)).")

decode_action(game::AbstractGame, player::Int, index::Int) =
    error("decode_action not implemented for $(typeof(game)).")

"""
Minimal hot-path transition.

At decision nodes:
    `action` is a player action.

At simultaneous nodes:
    `action` should be `JointAction{N}` for generic fixed-player tooling.

At chance nodes:
    `action` is `SampleChance()` or `ChanceOutcome(event)`.

Returns:
    next_state, rewards, terminated
"""
step(game::AbstractGame, state, action, rng::AbstractRNG = Random.default_rng()) =
    error("step not implemented for $(typeof(game)).")

default_info() = NamedTuple()

"""
Optional richer stepping interface.

Recommendation:
- return a `NamedTuple` for `info` to preserve usability and type stability
"""
function step_with_info(game::AbstractGame, state, action, rng::AbstractRNG = Random.default_rng())
    next_state, rewards, terminated = step(game, state, action, rng)
    return next_state, rewards, terminated, default_info()
end

"""
Canonical post-state observation interface.

The kernel contract is state-based:
    observe(game, next_state, player)

Games whose emitted observations depend on the transition itself may additionally
implement the optional exact-layer hook:
    observe_transition(game, state, action, next_state, player)
"""
observe(game::AbstractGame, state, player::Int) =
    error("observe not implemented for $(typeof(game)).")

is_terminal(game::AbstractGame, state) = node_kind(game, state) == TERMINAL

# ------------------------------------------------------------------
# Validation helpers for tighter indexed-action contracts
# ------------------------------------------------------------------

function validate_legal_action_mask(game::AbstractGame, state, player::Int)
    mask = legal_action_mask(game, state, player)
    n = num_base_actions(game, player)

    length(mask) == n ||
        throw(ArgumentError("legal_action_mask length $(length(mask)) does not match num_base_actions=$n for player $player."))

    for i in eachindex(mask)
        v = mask[i]
        (v isa Bool || v == 0 || v == 1) ||
            throw(ArgumentError("legal_action_mask must contain boolean-like values; got element $(repr(v)) at index $i."))
    end

    return mask
end

function validate_indexed_action_contract(game::AbstractGame, state, player::Int)
    action_mode(typeof(game)) === IndexedActions || return nothing

    legal = legal_actions(game, state, player)
    _ = validate_legal_action_mask(game, state, player)

    for a in legal
        (a isa Integer && 1 <= a <= num_base_actions(game, player)) ||
            throw(ArgumentError("IndexedActions legal_actions must return valid integer action indices; got $(repr(a))."))
    end

    return nothing
end

end