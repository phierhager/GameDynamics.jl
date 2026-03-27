using StaticArrays
using Base.Threads
using Statistics

export AbstractGame, AbstractSimultaneousGame, AbstractSequentialGame
export CHANCE_PLAYER
export player_to_move, legal_actions, transition, reward, observe, is_terminal, chance_outcomes
export act, learn, simulate_episode, run_episodes!, evaluate_parallel

# ---------------------------------------------------------------------------
# 1. Core Types and Constants
# ---------------------------------------------------------------------------
abstract type AbstractGame end
abstract type AbstractSimultaneousGame <: AbstractGame end
abstract type AbstractSequentialGame <: AbstractGame end

"""
Reserved player ID for environment stochasticity (dice rolls, card draws).
"""
const CHANCE_PLAYER = 0

# ---------------------------------------------------------------------------
# 2. The Expanded Game API
# ---------------------------------------------------------------------------
function player_to_move end  # (game, s) -> Int (1..N, or CHANCE_PLAYER)
function legal_actions end   # (game, s, p) -> valid actions for player `p`
function transition end      # (game, s, action) -> next_state
function reward end          # (game, s, action, s_next) -> NTuple of rewards
function observe end         # (game, s, i) -> info_set
function is_terminal end     # (game, s) -> Bool

"""
    chance_outcomes(game, s)

Required if `player_to_move(game, s) == CHANCE_PLAYER`.
Returns an iterable (Tuple or SVector to prevent allocation) of `(action, probability)` pairs.
"""
function chance_outcomes end

"""
    counterfactual_rewards(game, player_id, joint_action)

Returns an iterable of rewards `player_id` WOULD have received for every possible 
action, assuming opponents' actions remained fixed. 

Currently only supported for finite, simultaneous games.
"""
function counterfactual_rewards(game::AbstractGame, p::Int, a_joint)
    error("counterfactual_rewards is not defined for $(typeof(game)). Regret Matching requires finite, simultaneous games (like NormalFormGame).")
end

# ---------------------------------------------------------------------------
# 3. The Context-Aware Agent Protocol
# ---------------------------------------------------------------------------

"""
Agents now receive the `game` object. 
Model-free agents can ignore it. Model-based agents (MCTS) can use it to simulate.
"""
function act(agent, game, obs, valid_actions)
    error("Not implemented for $(typeof(agent))")
end

function learn(agent, game, obs, action, reward, next_obs)
    return agent # Default fallback for non-learning/frozen agents
end


"""
A zero-allocation continuous space wrapper.
Both `low` and `high` must be SVectors of the same dimension and type.
"""
struct ContinuousSpace{N, T}
    low::SVector{N, T}
    high::SVector{N, T}
end

function Base.rand(space::ContinuousSpace{N, T}) where {N, T}
    return space.low .+ rand(SVector{N, T}) .* (space.high .- space.low)
end

function clip(action::SVector{N, T}, space::ContinuousSpace{N, T}) where {N, T}
    return clamp.(action, space.low, space.high)
end

function Base.in(action::SVector, space::ContinuousSpace)
    return all(action .>= space.low) && all(action .<= space.high)
end