using Random

# ============================================================================
# Exports
# ============================================================================

export AbstractGame, AbstractState
export GameSpec, RewardSemantics, STEP_REWARDS, TERMINAL_RETURNS, MIXED_RETURNS

export AbstractNodeType, DecisionNode, SimultaneousNode, ChanceNode, TerminalNode
export DECISION_NODE, SIMULTANEOUS_NODE, CHANCE_NODE, TERMINAL_NODE

export AbstractSpace, DiscreteSpace, ContinuousSpace, BoxSpace, SimplexSpace, ProductSpace
export contains, sample, space_ndims

export JointAction, joint_action, players

export TransitionRecord, ChanceRecord, EpisodeRecord

export game_spec, num_players, player_ids, utility_bounds
export new_initial_state, node_type, current_player, active_players

export action_space, legal_actions, action_mask
export observation, information_state, public_observation
export observation_space, information_state_space, public_observation_space

export chance_outcomes, sample_chance
export apply_action, apply_actions

export rewards, returns, reward_semantics
export render, clone_state

export zero_rewards, as_reward_vector
export is_terminal, is_decision_node, is_simultaneous_node, is_chance_node

export counterfactual_values

# ============================================================================
# Core Abstract Types
# ============================================================================

abstract type AbstractGame end
abstract type AbstractState end

abstract type AbstractNodeType end
struct DecisionNode    <: AbstractNodeType end
struct SimultaneousNode <: AbstractNodeType end
struct ChanceNode      <: AbstractNodeType end
struct TerminalNode    <: AbstractNodeType end

const DECISION_NODE    = DecisionNode()
const SIMULTANEOUS_NODE = SimultaneousNode()
const CHANCE_NODE      = ChanceNode()
const TERMINAL_NODE    = TerminalNode()

# ============================================================================
# Reward Contract
# ============================================================================

"""
Reward semantics contract for environments.

- STEP_REWARDS:
    `rewards(game, s, a, s′)` returns the full incremental reward emitted on each transition.
    `returns(game, sT)` is optional metadata / summary and is NOT accumulated by the engine.

- TERMINAL_RETURNS:
    `rewards(...)` should be zero (or omitted conceptually) for all transitions.
    `returns(game, sT)` defines the terminal utility vector.
    The engine should add it once when termination is reached.

- MIXED_RETURNS:
    `rewards(...)` are incremental transition rewards and
    `returns(game, sT)` is an additional terminal utility vector added once at the end.
"""
abstract type RewardSemantics end
struct StepRewardsSemantics <: RewardSemantics end
struct TerminalReturnsSemantics <: RewardSemantics end
struct MixedReturnsSemantics <: RewardSemantics end

const STEP_REWARDS     = StepRewardsSemantics()
const TERMINAL_RETURNS = TerminalReturnsSemantics()
const MIXED_RETURNS    = MixedReturnsSemantics()

# ============================================================================
# Spaces API
# ============================================================================

abstract type AbstractSpace end

"""
Finite, explicit discrete action/observation space.
"""
struct DiscreteSpace{T} <: AbstractSpace
    elements::Vector{T}
end

"""
Axis-aligned continuous box in R^n.
"""
struct BoxSpace{T, V<:AbstractVector{T}} <: AbstractSpace
    low::V
    high::V
end

"""
Alias for a box-valued action/observation space.
"""
const ContinuousSpace = BoxSpace

"""
Probability simplex of dimension n:
    {x in R^n : x_i >= 0, sum(x) = 1}
"""
struct SimplexSpace <: AbstractSpace
    n::Int
end

"""
Cartesian product of spaces.
"""
struct ProductSpace{S<:Tuple} <: AbstractSpace
    spaces::S
end

space_ndims(::DiscreteSpace) = 1
space_ndims(s::BoxSpace) = length(s.low)
space_ndims(s::SimplexSpace) = s.n
space_ndims(s::ProductSpace) = sum(space_ndims(x) for x in s.spaces)

contains(space::DiscreteSpace, x) = x in space.elements
contains(space::BoxSpace, x) = length(x) == length(space.low) && all(space.low .<= x .<= space.high)
contains(space::SimplexSpace, x) = length(x) == space.n && all(x .>= 0) && isapprox(sum(x), 1.0; atol=1e-8)
contains(space::ProductSpace, x) = length(x) == length(space.spaces) &&
                                   all(contains(space.spaces[i], x[i]) for i in eachindex(space.spaces))

function sample(rng::AbstractRNG, space::DiscreteSpace)
    isempty(space.elements) && throw(ArgumentError("Cannot sample from empty DiscreteSpace."))
    return rand(rng, space.elements)
end

function sample(rng::AbstractRNG, space::BoxSpace)
    length(space.low) == length(space.high) || throw(ArgumentError("BoxSpace low/high mismatch."))
    return space.low .+ rand(rng, length(space.low)) .* (space.high .- space.low)
end

function sample(rng::AbstractRNG, space::SimplexSpace)
    x = rand(rng, space.n)
    return x ./ sum(x)
end

function sample(rng::AbstractRNG, space::ProductSpace)
    return tuple((sample(rng, s) for s in space.spaces)...)
end

# ============================================================================
# Game Metadata
# ============================================================================

Base.@kwdef struct GameSpec
    num_players::Int
    perfect_information::Bool = true
    perfect_recall::Bool = true
    stochastic::Bool = false
    simultaneous_moves::Bool = false
    zero_sum::Bool = false
    general_sum::Bool = false

    max_episode_length::Union{Nothing, Int} = nothing
    utility_bounds::Union{Nothing, Tuple{Float64, Float64}} = nothing

    observation_mode::Symbol = :state
    provides_information_state::Bool = false
    provides_public_observation::Bool = false
    provides_action_mask::Bool = false

    reward_semantics::RewardSemantics = STEP_REWARDS
end

# ============================================================================
# Joint Actions
# ============================================================================

"""
Canonical representation of simultaneous actions keyed by player id.
"""
struct JointAction
    actions::Dict{Int, Any}
end

joint_action(actions::Dict{Int, Any}) = JointAction(Dict(actions))
joint_action(actions::AbstractDict)   = JointAction(Dict(Int(k) => v for (k, v) in pairs(actions)))
joint_action(actions::AbstractVector) = JointAction(Dict(i => actions[i] for i in eachindex(actions)))
joint_action(actions::Tuple)          = JointAction(Dict(i => actions[i] for i in 1:length(actions)))

Base.getindex(a::JointAction, player::Int) = a.actions[player]
Base.length(a::JointAction) = length(a.actions)
Base.haskey(a::JointAction, player::Int) = haskey(a.actions, player)
Base.iterate(a::JointAction, state...) = iterate(a.actions, state...)
Base.pairs(a::JointAction) = pairs(a.actions)
players(a::JointAction) = sort!(collect(keys(a.actions)))

# ============================================================================
# Canonical Transition / Episode Records
# ============================================================================

"""
Canonical environment transition record.

This is the main object that downstream trainers/solvers should consume.

Fields:
- `state`: pre-transition state
- `node`: node type before acting
- `acting_players`: players allowed/required to act at `state`
- `action`: action taken (scalar action, joint action, chance action, or `nothing`)
- `next_state`: successor state
- `rewards`: incremental transition rewards only
- `terminated`: whether `next_state` is terminal
- `truncated`: whether the episode was truncated by horizon/driver
- `chance_prob`: probability of sampled chance action, if known
- `info`: free-form metadata
"""
struct TransitionRecord{S, N<:AbstractNodeType, A, NS, I}
    state::S
    node::N
    acting_players::Vector{Int}
    action::A
    next_state::NS
    rewards::Vector{Float64}
    terminated::Bool
    truncated::Bool
    chance_prob::Union{Nothing, Float64}
    info::I
end

"""
Optional specialized record for explicit chance transitions.
"""
struct ChanceRecord{S, A, NS, I}
    state::S
    action::A
    probability::Float64
    next_state::NS
    rewards::Vector{Float64}
    terminated::Bool
    info::I
end

"""
Canonical episode summary record.
`cumulative_rewards` contains only accumulated transition rewards.
`terminal_returns` contains the result of `returns(game, terminal_state)` if queried by the driver.
`payoffs` is the final total actually used by the driver after applying reward semantics.
"""
struct EpisodeRecord{S}
    initial_state::S
    terminal_state::S
    transitions::Vector{TransitionRecord}
    cumulative_rewards::Vector{Float64}
    terminal_returns::Vector{Float64}
    payoffs::Vector{Float64}
    steps::Int
    terminated::Bool
    truncated::Bool
end

# ============================================================================
# Game Introspection
# ============================================================================

function game_spec(game::AbstractGame)
    error("game_spec is not implemented for $(typeof(game)).")
end

num_players(game::AbstractGame) = game_spec(game).num_players
player_ids(game::AbstractGame) = 1:num_players(game)
utility_bounds(game::AbstractGame) = game_spec(game).utility_bounds
reward_semantics(game::AbstractGame) = game_spec(game).reward_semantics

# ============================================================================
# State / Node API
# ============================================================================

function new_initial_state(game::AbstractGame, rng::AbstractRNG = Random.default_rng())
    error("new_initial_state is not implemented for $(typeof(game)).")
end

function node_type(game::AbstractGame, state)
    error("node_type is not implemented for $(typeof(game)) and state $(typeof(state)).")
end

function current_player(game::AbstractGame, state)
    error("current_player is only defined on decision nodes for $(typeof(game)).")
end

function active_players(game::AbstractGame, state)
    error("active_players is only defined on simultaneous nodes for $(typeof(game)).")
end

# ============================================================================
# Action API
# ============================================================================

"""
Returns the action space available to `player` at `state`.

For discrete games this will usually be a `DiscreteSpace`.
For continuous or structured games it may be `BoxSpace`, `SimplexSpace`, or `ProductSpace`.
"""
function action_space(game::AbstractGame, state, player::Int)
    legal = legal_actions(game, state, player)
    if legal isa AbstractSpace
        return legal
    elseif legal isa Tuple || legal isa AbstractVector || legal isa AbstractRange
        return DiscreteSpace(collect(legal))
    else
        error("action_space is not implemented for $(typeof(game)), and legal_actions does not define a discrete collection.")
    end
end

"""
Returns the legal actions for `player` at `state`.

Contract:
- For discrete games: return an iterable of actions.
- For continuous/structured games: may return an `AbstractSpace`.
"""
function legal_actions(game::AbstractGame, state, player::Int)
    error("legal_actions is not implemented for $(typeof(game)) and player $player.")
end

"""
Optional boolean mask aligned with a canonical discrete action encoding.
Returns `nothing` unless the environment provides one.
"""
action_mask(game::AbstractGame, state, player::Int) = nothing

# ============================================================================
# Observation / Information API
# ============================================================================

"""
Per-player observation emitted by the environment.
May be full state, private observation, or feature view.
"""
observation(game::AbstractGame, state, player::Int) = state

"""
Per-player information state for imperfect-information algorithms.
Default: same as observation.
"""
information_state(game::AbstractGame, state, player::Int) = observation(game, state, player)

"""
Public observation shared across players.
Default: `nothing`.
"""
public_observation(game::AbstractGame, state) = nothing

"""
Observation space for player `player`.
Default: `nothing` if not declared.
"""
observation_space(game::AbstractGame, player::Int) = nothing

"""
Information-state space for player `player`.
Default: `nothing` if not declared.
"""
information_state_space(game::AbstractGame, player::Int) = nothing

"""
Public observation space.
Default: `nothing` if not declared.
"""
public_observation_space(game::AbstractGame) = nothing

# ============================================================================
# Chance API
# ============================================================================

"""
Returns an iterable of `(action, probability)` pairs for chance nodes.
"""
function chance_outcomes(game::AbstractGame, state)
    error("chance_outcomes is not implemented for $(typeof(game)).")
end

function sample_chance(game::AbstractGame, state, rng::AbstractRNG = Random.default_rng())
    outcomes = chance_outcomes(game, state)
    r = rand(rng)
    cumulative = 0.0
    fallback = nothing
    for (action, prob) in outcomes
        fallback = action
        cumulative += prob
        if r <= cumulative + eps(Float64)
            return action
        end
    end
    fallback === nothing && error("chance_outcomes returned an empty iterable for $(typeof(game)).")
    return fallback
end

# ============================================================================
# Transition Dynamics API
# ============================================================================

function apply_action(game::AbstractGame, state, action)
    error("apply_action is not implemented for $(typeof(game)).")
end

function apply_actions(game::AbstractGame, state, joint::JointAction)
    error("apply_actions is not implemented for $(typeof(game)).")
end

# ============================================================================
# Reward / Return API
# ============================================================================

"""
Incremental reward emitted on the transition `(state, action, next_state)`.

Contract:
- Must return only *transition-local* rewards.
- Must NOT include whole-episode cumulative returns unless the game explicitly uses `STEP_REWARDS`
  and intends those to be emitted as a single terminal step reward.
- The engine accumulates `rewards(...)` over time.
"""
function rewards(game::AbstractGame, state, action, next_state)
    return zero_rewards(game)
end

"""
Terminal utility summary for `terminal_state`.

Contract:
- Only meaningful when `reward_semantics(game)` is `TERMINAL_RETURNS` or `MIXED_RETURNS`.
- Must represent terminal utility only, not transition-local reward already emitted by `rewards(...)`.
- Should be queried exactly once by the driver upon termination.
"""
function returns(game::AbstractGame, terminal_state)
    return zero_rewards(game)
end

# ============================================================================
# Utilities / Rendering
# ============================================================================

render(game::AbstractGame, state; mode::Symbol = :text) = string(state)
clone_state(state) = deepcopy(state)

# ============================================================================
# Solver / Deviations API
# ============================================================================

function counterfactual_values(game::AbstractGame, state, player::Int, joint::JointAction)
    error("counterfactual_values is not implemented for $(typeof(game)).")
end

# ============================================================================
# Reward Helpers
# ============================================================================

zero_rewards(game::AbstractGame) = zeros(Float64, num_players(game))

function as_reward_vector(game::AbstractGame, r)
    n = num_players(game)

    if r isa Real
        n == 1 || throw(ArgumentError("Scalar rewards are only valid for single-player games."))
        return [Float64(r)]

    elseif r isa Tuple
        length(r) == n || throw(ArgumentError("Expected $n rewards, got $(length(r))."))
        return [Float64(x) for x in r]

    elseif r isa AbstractVector
        length(r) == n || throw(ArgumentError("Expected $n rewards, got $(length(r))."))
        return Float64.(collect(r))

    else
        throw(ArgumentError("Rewards must be a scalar, tuple, or vector; got $(typeof(r))."))
    end
end

# ============================================================================
# Node Predicates
# ============================================================================

is_terminal(game::AbstractGame, state) = node_type(game, state) isa TerminalNode
is_decision_node(game::AbstractGame, state) = node_type(game, state) isa DecisionNode
is_simultaneous_node(game::AbstractGame, state) = node_type(game, state) isa SimultaneousNode
is_chance_node(game::AbstractGame, state) = node_type(game, state) isa ChanceNode