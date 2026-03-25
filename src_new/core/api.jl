module CoreAPI

export AbstractGame, AbstractState
export NodeType, DecisionNode, ChanceNode, SimultaneousNode, TerminalNode
export new_initial_state, node_type, current_player, active_players
export legal_actions, chance_outcomes, apply_action!, apply_actions!
export observation, information_state, public_observation
export returns, rewards, is_terminal, clone

# --- 1. Core Types ---

abstract type AbstractGame end
abstract type AbstractState end

# Node Types for the State Machine
abstract type NodeType end
struct DecisionNode <: NodeType end
struct ChanceNode <: NodeType end
struct SimultaneousNode <: NodeType end
struct TerminalNode <: NodeType end

# --- 2. Game Metadata & Initialization ---

"""
    new_initial_state(game::AbstractGame, rng::AbstractRNG) -> AbstractState

Creates a fresh, mutable state object for the beginning of the game.
"""
function new_initial_state end

# --- 3. State Machine Navigation ---

"""
    node_type(state::AbstractState) -> NodeType

Returns the type of the current node (Decision, Chance, Simultaneous, Terminal).
"""
function node_type end

"""
    current_player(state::AbstractState) -> Int

Returns the ID of the player to act. Valid only at DecisionNodes.
"""
function current_player end

"""
    active_players(state::AbstractState) -> Vector{Int}

Returns a list of players who must act. Valid only at SimultaneousNodes.
"""
function active_players end

# --- 4. Actions & Transitions ---

"""
    legal_actions(state::AbstractState, player::Int) -> Vector{ActionType}

Returns the allowed actions for `player` at the current state.
"""
function legal_actions end

"""
    chance_outcomes(state::AbstractState) -> Vector{Tuple{ActionType, Float64}}

Returns a list of (outcome, probability) pairs. Valid only at ChanceNodes.
"""
function chance_outcomes end

"""
    apply_action!(state::AbstractState, action)

Mutates the state by applying a single action. Used for Decision and Chance nodes.
"""
function apply_action! end

"""
    apply_actions!(state::AbstractState, joint_action::Dict{Int, ActionType})

Mutates the state by applying actions from multiple players. Used for Simultaneous nodes.
"""
function apply_actions! end

# --- 5. Observations & Information (The crucial split) ---

"""
    observation(state::AbstractState, player::Int) -> Array/String

Returns the current, potentially partial, view of the state for the given player.
"""
function observation end

"""
    information_state(state::AbstractState, player::Int) -> Array/String

Returns the perfect-recall history/information state for the given player. 
Crucial for imperfect-information algorithms like CFR.
"""
function information_state end

"""
    public_observation(state::AbstractState) -> Array/String

Returns the information that is universally known to all players.
"""
function public_observation end

# --- 6. Rewards & Returns ---

"""
    rewards(state::AbstractState) -> Vector{Float64}

Returns the step-rewards for all players resulting from the last transition.
"""
function rewards end

"""
    returns(state::AbstractState) -> Vector{Float64}

Returns the cumulative / terminal returns for all players. Valid only at TerminalNodes.
"""
function returns end

"""
    is_terminal(state::AbstractState) -> Bool
"""
function is_terminal end

# --- 7. Utilities ---

"""
    clone(state::AbstractState) -> AbstractState

Creates a deep copy of the state. Essential for tree-search (MCTS) and planning.
"""
function clone end

end # module