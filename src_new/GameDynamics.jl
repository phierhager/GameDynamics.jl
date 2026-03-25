using Random

# ============================================================================
# Exports
# ============================================================================

export AbstractGame, AbstractState

export GameSpec, CapabilitySpec, HorizonSpec
export AbstractNodeType, DecisionNode, SimultaneousNode, ChanceNode, TerminalNode
export DECISION_NODE, SIMULTANEOUS_NODE, CHANCE_NODE, TERMINAL_NODE

export AbstractTimeModel, DiscreteTimeModel, ContinuousTimeModel, EventDrivenTimeModel
export time_model, time_index, initial_time, time_step_kind, elapsed_time

export AbstractSpace, FiniteSpace, BoxSpace, ContinuousSpace, SimplexSpace, ProductSpace, HybridSpace
export state_space, action_space, observation_space, information_state_space, public_observation_space, signal_space
export contains, sample, dimension

export AbstractPlayerModel, FixedPlayers, DynamicPlayers, PopulationPlayers
export PlayerRole, Team, Coalition, MechanismSpec
export player_model, num_players, player_ids, active_player_ids, role_of, roles, teams, coalitions, mechanism_spec

export AbstractHistory, NullHistory, DecisionEvent, SimultaneousEvent, ChanceEvent, SignalEvent
export history_type, initial_history, append_event, decision_history, info_set_id, public_state

export JointAction, joint_action, players

export AbstractExogenousModel, ExplicitChanceModel, SampleOnlyExogenousModel, LatentProcessModel
export exogenous_model, chance_outcomes, sample_exogenous, exogenous_density, exogenous_support

export feasible_actions, legal_actions, action_constraints, action_mask
export observation, information_state, public_observation, signals, recommendations

export AbstractUtility, UtilityVector, ScalarUtility, LexicographicUtility, DistributionalUtility
export utility_semantics, utility_bounds, rewards, returns, stage_utility, terminal_utility
export zero_utility, utility_add, utility_scale, as_utility

export apply_action, apply_actions, transition_kernel, observation_kernel
export clone_state, render

export TransitionRecord, EpisodeRecord

export is_terminal, is_decision_node, is_simultaneous_node, is_chance_node
export node_type, current_player, active_players
export new_initial_state
export counterfactual_values

# ============================================================================
# Core Abstract Types
# ============================================================================

abstract type AbstractGame end
abstract type AbstractState end

# ============================================================================
# Node Types
# ============================================================================

abstract type AbstractNodeType end
struct DecisionNode <: AbstractNodeType end
struct SimultaneousNode <: AbstractNodeType end
struct ChanceNode <: AbstractNodeType end
struct TerminalNode <: AbstractNodeType end

const DECISION_NODE     = DecisionNode()
const SIMULTANEOUS_NODE = SimultaneousNode()
const CHANCE_NODE       = ChanceNode()
const TERMINAL_NODE     = TerminalNode()

# ============================================================================
# Time Model
# ============================================================================

abstract type AbstractTimeModel end

"""
Classic stage-based / turn-based games.
"""
struct DiscreteTimeModel{T<:Integer} <: AbstractTimeModel
    initial::T
    dt::T
end

"""
Continuous-time games, differential games, stochastic differential games.
The API does not force a solver; it only exposes the semantics.
"""
struct ContinuousTimeModel{T<:Real} <: AbstractTimeModel
    initial::T
end

"""
Event-driven model. Time advances by event durations or jump times.
Useful for semi-Markov games, queueing games, asynchronous games, etc.
"""
struct EventDrivenTimeModel{T<:Real} <: AbstractTimeModel
    initial::T
end

initial_time(m::DiscreteTimeModel) = m.initial
initial_time(m::ContinuousTimeModel) = m.initial
initial_time(m::EventDrivenTimeModel) = m.initial

"""
Returns the time model for the game.
"""
function time_model(game::AbstractGame)
    return DiscreteTimeModel(0, 1)
end

"""
Returns the current time index / timestamp encoded in state.
Games should override if time is part of the state semantics.
"""
time_index(game::AbstractGame, state) = nothing

"""
Returns a symbolic description of how time advances at the current state:
- :fixed_step
- :control_interval
- :jump
- :terminal
"""
function time_step_kind(game::AbstractGame, state)
    node = node_type(game, state)
    return node isa TerminalNode ? :terminal : :fixed_step
end

"""
Returns elapsed time for the transition (state, action, next_state).

For discrete games, default is 1.
For continuous / event-driven games, environments should override.
"""
function elapsed_time(game::AbstractGame, state, action, next_state)
    tm = time_model(game)
    if tm isa DiscreteTimeModel
        return tm.dt
    else
        return 0.0
    end
end

# ============================================================================
# Space API
# ============================================================================

abstract type AbstractSpace end

"""
Finite explicit space. Use tuples or static vectors in concrete envs if desired.
"""
struct FiniteSpace{T} <: AbstractSpace
    elements::Vector{T}
end

"""
Axis-aligned box in R^n.
"""
struct BoxSpace{T<:Real, V<:AbstractVector{T}} <: AbstractSpace
    low::V
    high::V
end

const ContinuousSpace = BoxSpace

"""
n-dimensional probability simplex.
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

"""
Hybrid structured space, for example discrete + continuous parameter.
"""
struct HybridSpace{A<:AbstractSpace, B<:AbstractSpace} <: AbstractSpace
    discrete_part::A
    continuous_part::B
end

dimension(::FiniteSpace) = 1
dimension(s::BoxSpace) = length(s.low)
dimension(s::SimplexSpace) = s.n
dimension(s::ProductSpace) = sum(dimension(x) for x in s.spaces)
dimension(s::HybridSpace) = dimension(s.discrete_part) + dimension(s.continuous_part)

contains(space::FiniteSpace, x) = x in space.elements
contains(space::BoxSpace, x) = length(x) == length(space.low) && all(space.low .<= x .<= space.high)
contains(space::SimplexSpace, x) = length(x) == space.n && all(x .>= 0) && isapprox(sum(x), 1.0; atol=1e-8)
contains(space::ProductSpace, x) = length(x) == length(space.spaces) &&
                                   all(contains(space.spaces[i], x[i]) for i in eachindex(space.spaces))
contains(space::HybridSpace, x) = length(x) == 2 &&
                                  contains(space.discrete_part, x[1]) &&
                                  contains(space.continuous_part, x[2])

function sample(rng::AbstractRNG, space::FiniteSpace)
    isempty(space.elements) && throw(ArgumentError("Cannot sample from empty FiniteSpace."))
    return rand(rng, space.elements)
end

function sample(rng::AbstractRNG, space::BoxSpace)
    return space.low .+ rand(rng, length(space.low)) .* (space.high .- space.low)
end

function sample(rng::AbstractRNG, space::SimplexSpace)
    x = rand(rng, space.n)
    return x ./ sum(x)
end

function sample(rng::AbstractRNG, space::ProductSpace)
    return tuple((sample(rng, s) for s in space.spaces)...)
end

function sample(rng::AbstractRNG, space::HybridSpace)
    return (sample(rng, space.discrete_part), sample(rng, space.continuous_part))
end

# ============================================================================
# Player / Role / Team / Coalition / Mechanism API
# ============================================================================

abstract type AbstractPlayerModel end

"""
Fixed finite players, classic normal-form / extensive-form setting.
"""
struct FixedPlayers <: AbstractPlayerModel
    ids::UnitRange{Int}
end

"""
Players may enter/exit over time.
"""
struct DynamicPlayers <: AbstractPlayerModel
    initial_ids::Vector{Int}
end

"""
Population / mean-field / anonymous / nonatomic approximation model.
`roles` may define populations like :buyers, :sellers, :representative_agent.
"""
struct PopulationPlayers <: AbstractPlayerModel
    role_names::Vector{Symbol}
end

struct PlayerRole
    player::Int
    role::Symbol
end

struct Team
    id::Symbol
    members::Vector{Int}
end

struct Coalition
    id::Symbol
    members::Vector{Int}
    transferable_utility::Bool
end

"""
Mechanism / mediator / institution metadata.
"""
Base.@kwdef struct MechanismSpec
    has_public_recommendations::Bool = false
    has_private_recommendations::Bool = false
    has_cheap_talk::Bool = false
    has_binding_commitments::Bool = false
    has_transfers::Bool = false
end

function player_model(game::AbstractGame)
    return FixedPlayers(1:1)
end

function num_players(game::AbstractGame)
    pm = player_model(game)
    if pm isa FixedPlayers
        return length(pm.ids)
    elseif pm isa DynamicPlayers
        return length(pm.initial_ids)
    elseif pm isa PopulationPlayers
        return length(pm.role_names)
    else
        error("num_players not defined for player model $(typeof(pm)).")
    end
end

function player_ids(game::AbstractGame)
    pm = player_model(game)
    if pm isa FixedPlayers
        return pm.ids
    elseif pm isa DynamicPlayers
        return pm.initial_ids
    elseif pm isa PopulationPlayers
        return 1:length(pm.role_names)
    else
        error("player_ids not defined for player model $(typeof(pm)).")
    end
end

"""
Players currently alive / present in the game.
Override for dynamic-population games.
"""
active_player_ids(game::AbstractGame, state) = collect(player_ids(game))

role_of(game::AbstractGame, player::Int) = Symbol("player_", player)
roles(game::AbstractGame) = [role_of(game, p) for p in player_ids(game)]
teams(game::AbstractGame) = Team[]
coalitions(game::AbstractGame) = Coalition[]
mechanism_spec(game::AbstractGame) = MechanismSpec()

# ============================================================================
# History / Information-Set API
# ============================================================================

abstract type AbstractHistory end

struct NullHistory <: AbstractHistory end

struct DecisionEvent{T,A}
    t::T
    player::Int
    action::A
end

struct SimultaneousEvent{T,A}
    t::T
    acting_players::Vector{Int}
    joint_action::A
end

struct ChanceEvent{T,A}
    t::T
    outcome::A
    probability::Union{Nothing, Float64}
end

struct SignalEvent{T,S}
    t::T
    sender::Union{Nothing, Int, Symbol}
    receiver::Union{Nothing, Int, Symbol}
    signal::S
    public::Bool
end

history_type(game::AbstractGame) = NullHistory
initial_history(game::AbstractGame) = NullHistory()

"""
Append an event to a game history. Concrete envs with efficient history structs
should override this.
"""
append_event(game::AbstractGame, h::NullHistory, e) = (e,)

function append_event(game::AbstractGame, h::Tuple, e)
    return (h..., e)
end

"""
Decision-relevant history. By default same as full history.
Override to compress or canonicalize.
"""
decision_history(game::AbstractGame, state) = nothing

"""
Information-set identifier for algorithms like CFR.
Must be player-specific in imperfect-information games.
"""
info_set_id(game::AbstractGame, state, player::Int) = nothing

"""
Public state abstraction for public-tree or public-belief algorithms.
"""
public_state(game::AbstractGame, state) = public_observation(game, state)

# ============================================================================
# Joint Action
# ============================================================================

struct JointAction{A}
    actions::A
end

joint_action(actions) = JointAction(actions)

Base.getindex(a::JointAction, player::Int) = a.actions[player]
Base.length(a::JointAction) = length(a.actions)
players(a::JointAction) = collect(keys(a.actions))

# ============================================================================
# Exogenous / Chance API
# ============================================================================

abstract type AbstractExogenousModel end

"""
Explicit enumerable stochastic process.
"""
struct ExplicitChanceModel <: AbstractExogenousModel end

"""
Sampling only; exact enumeration may be impossible or undesirable.
"""
struct SampleOnlyExogenousModel <: AbstractExogenousModel end

"""
Latent exogenous process with hidden state evolution.
"""
struct LatentProcessModel <: AbstractExogenousModel end

exogenous_model(game::AbstractGame) = ExplicitChanceModel()

"""
Exact support of exogenous events if available.
"""
function exogenous_support(game::AbstractGame, state)
    return nothing
end

"""
Explicit enumerable outcomes `(event, probability)` if available.
Equivalent to classic chance outcomes.
"""
function chance_outcomes(game::AbstractGame, state)
    error("chance_outcomes is not implemented for $(typeof(game)).")
end

"""
More general exogenous sampling entry point.
Can be used for chance nodes, latent shocks, jump times, noise draws, etc.
"""
function sample_exogenous(game::AbstractGame, state, rng::AbstractRNG = Random.default_rng())
    model = exogenous_model(game)
    if model isa ExplicitChanceModel
        outcomes = chance_outcomes(game, state)
        r = rand(rng)
        cumulative = 0.0
        fallback = nothing
        for (event, prob) in outcomes
            fallback = event
            cumulative += prob
            if r <= cumulative + eps(Float64)
                return event
            end
        end
        fallback === nothing && error("chance_outcomes returned an empty iterable for $(typeof(game)).")
        return fallback
    else
        error("sample_exogenous is not implemented for $(typeof(game)).")
    end
end

"""
Optional exact density / probability mass for an exogenous event.
Needed by some planners / filters.
"""
function exogenous_density(game::AbstractGame, state, event)
    return nothing
end

# ============================================================================
# Utility Semantics
# ============================================================================

abstract type AbstractUtility end

"""
Standard vector-valued per-player utilities.
"""
struct UtilityVector{T<:Real, V<:AbstractVector{T}} <: AbstractUtility
    values::V
end

"""
Single-agent or single-objective scalar utility.
"""
struct ScalarUtility{T<:Real} <: AbstractUtility
    value::T
end

"""
Lexicographic / hierarchical objectives.
"""
struct LexicographicUtility{T<:Real} <: AbstractUtility
    levels::Vector{Vector{T}}
end

"""
Distribution-valued utility / return object.
"""
struct DistributionalUtility{T} <: AbstractUtility
    support::Vector{T}
    probs::Vector{Float64}
end

"""
Defines how utilities are represented by the game.
"""
utility_semantics(game::AbstractGame) = UtilityVector

"""
Bounds for utility objects if meaningful.
"""
utility_bounds(game::AbstractGame) = nothing

function zero_utility(game::AbstractGame)
    return UtilityVector(zeros(Float64, num_players(game)))
end

function utility_add(u1::UtilityVector, u2::UtilityVector)
    return UtilityVector(u1.values .+ u2.values)
end

function utility_scale(a::Real, u::UtilityVector)
    return UtilityVector(a .* u.values)
end

function as_utility(game::AbstractGame, x)
    if x isa AbstractUtility
        return x
    elseif x isa Real
        num_players(game) == 1 || throw(ArgumentError("Scalar utility only valid for single-player games unless game overrides as_utility."))
        return ScalarUtility(Float64(x))
    elseif x isa Tuple
        return UtilityVector(Float64.(collect(x)))
    elseif x isa AbstractVector
        return UtilityVector(Float64.(collect(x)))
    else
        throw(ArgumentError("Cannot convert $(typeof(x)) to utility object."))
    end
end

# ============================================================================
# Game and Capability Specs
# ============================================================================

Base.@kwdef struct HorizonSpec
    episodic::Bool = true
    finite_horizon::Bool = true
    max_steps::Union{Nothing, Int} = nothing
    max_time::Union{Nothing, Float64} = nothing
    discount::Union{Nothing, Float64} = nothing
    absorbing_terminal::Bool = true
end

Base.@kwdef struct CapabilitySpec
    provides_information_state::Bool = false
    provides_public_observation::Bool = false
    provides_action_mask::Bool = false
    provides_exact_chance_outcomes::Bool = false
    provides_transition_kernel::Bool = false
    provides_observation_kernel::Bool = false
    provides_counterfactual_values::Bool = false
    supports_continuous_time::Bool = false
    supports_dynamic_players::Bool = false
    supports_coalitions::Bool = false
    supports_signals::Bool = false
end

Base.@kwdef struct GameSpec
    perfect_information::Bool = true
    perfect_recall::Bool = true
    stochastic::Bool = false
    simultaneous_moves::Bool = false
    zero_sum::Bool = false
    general_sum::Bool = false
    horizon::HorizonSpec = HorizonSpec()
    capabilities::CapabilitySpec = CapabilitySpec()
end

function game_spec(game::AbstractGame)
    return GameSpec()
end

# ============================================================================
# State / Dynamics API
# ============================================================================

function new_initial_state(game::AbstractGame, rng::AbstractRNG = Random.default_rng())
    error("new_initial_state is not implemented for $(typeof(game)).")
end

function state_space(game::AbstractGame)
    return nothing
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
Base action domain, before state-dependent restrictions.
For structured continuous/discrete games this should be implemented.
"""
function action_space(game::AbstractGame, state, player::Int)
    legal = legal_actions(game, state, player)
    if legal isa AbstractSpace
        return legal
    elseif legal isa Tuple || legal isa AbstractVector || legal isa AbstractRange
        return FiniteSpace(collect(legal))
    else
        return nothing
    end
end

"""
Feasible actions: actions that satisfy low-level physical / type constraints.
These may still be strategically or institutionally illegal.
Default: same as action_space.
"""
feasible_actions(game::AbstractGame, state, player::Int) = action_space(game, state, player)

"""
Legal actions: actions admissible under the rules at this state.

Contract:
- for discrete games: iterable of concrete actions
- for continuous/hybrid games: may return an AbstractSpace
"""
function legal_actions(game::AbstractGame, state, player::Int)
    error("legal_actions is not implemented for $(typeof(game)) and player $player.")
end

"""
Optional constraint object or function for constrained action spaces.
Examples:
- simplex constraints
- budget constraints
- coupled constraints in continuous actions
- mechanism-induced admissibility constraints
"""
action_constraints(game::AbstractGame, state, player::Int) = nothing

"""
Optional discrete mask for algorithms that require one.
"""
action_mask(game::AbstractGame, state, player::Int) = nothing

# ============================================================================
# Observation / Information / Signal API
# ============================================================================

observation(game::AbstractGame, state, player::Int) = state
information_state(game::AbstractGame, state, player::Int) = observation(game, state, player)
public_observation(game::AbstractGame, state) = nothing

observation_space(game::AbstractGame, player::Int) = nothing
information_state_space(game::AbstractGame, player::Int) = nothing
public_observation_space(game::AbstractGame) = nothing

"""
Signals emitted at the current state or transition.
Used for cheap talk, mediator recommendations, public announcements, etc.
Return a concrete efficient container in envs.
"""
signals(game::AbstractGame, state) = ()

"""
Player-specific recommendations / prescriptions from a mediator or mechanism.
"""
recommendations(game::AbstractGame, state, player::Int) = nothing

signal_space(game::AbstractGame, state) = nothing

# ============================================================================
# Transition / Kernel API
# ============================================================================

"""
Single-actor transition.
Used on decision nodes and chance nodes.
"""
function apply_action(game::AbstractGame, state, action)
    error("apply_action is not implemented for $(typeof(game)).")
end

"""
Joint transition.
Used on simultaneous nodes.
"""
function apply_actions(game::AbstractGame, state, joint::JointAction)
    error("apply_actions is not implemented for $(typeof(game)).")
end

"""
Optional exact transition kernel:
returns an iterable of `(next_state, probability)` or equivalent kernel object.
Useful for exact planning / equilibrium solvers.
"""
transition_kernel(game::AbstractGame, state, action) = nothing

"""
Optional exact observation / signal kernel.
"""
observation_kernel(game::AbstractGame, state, action, next_state) = nothing

# ============================================================================
# Reward / Utility API
# ============================================================================

"""
Incremental stage utility emitted on a transition.
Can represent:
- standard per-player reward vectors,
- lexicographic signals,
- distributional utilities,
- coalition/team payoffs,
depending on the concrete utility object returned.
"""
function stage_utility(game::AbstractGame, state, action, next_state)
    return zero_utility(game)
end

"""
Terminal utility contribution. Queried once at termination if the driver uses it.
"""
function terminal_utility(game::AbstractGame, terminal_state)
    return zero_utility(game)
end

"""
Compatibility aliases.
"""
rewards(game::AbstractGame, state, action, next_state) = stage_utility(game, state, action, next_state)
returns(game::AbstractGame, terminal_state) = terminal_utility(game, terminal_state)

# ============================================================================
# Histories / Cloning / Rendering
# ============================================================================

clone_state(state) = deepcopy(state)
render(game::AbstractGame, state; mode::Symbol = :text) = string(state)

# ============================================================================
# Transition / Episode Records
# ============================================================================

"""
Canonical transition record.

This is intentionally generic and parametric. Performance-sensitive envs can
instantiate it with concrete state/history/utility/action types and avoid Any.
"""
struct TransitionRecord{S,N<:AbstractNodeType,A,NS,U,H,SG,I,T}
    state::S
    node::N
    acting_players::Vector{Int}
    action::A
    next_state::NS
    utility::U
    history::H
    emitted_signals::SG
    info::I
    dt::T
    terminated::Bool
    truncated::Bool
end

struct EpisodeRecord{S,U,H}
    initial_state::S
    terminal_state::S
    cumulative_utility::U
    terminal_utility::U
    realized_payoff::U
    history::H
    steps::Int
    terminated::Bool
    truncated::Bool
    elapsed_time::Any
end

# ============================================================================
# Derived Predicates
# ============================================================================

is_terminal(game::AbstractGame, state) = node_type(game, state) isa TerminalNode
is_decision_node(game::AbstractGame, state) = node_type(game, state) isa DecisionNode
is_simultaneous_node(game::AbstractGame, state) = node_type(game, state) isa SimultaneousNode
is_chance_node(game::AbstractGame, state) = node_type(game, state) isa ChanceNode

# ============================================================================
# Counterfactual / Solver API
# ============================================================================

function counterfactual_values(game::AbstractGame, state, player::Int, joint::JointAction)
    error("counterfactual_values is not implemented for $(typeof(game)).")
end