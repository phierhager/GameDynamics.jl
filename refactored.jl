# ===========================================================================
# GameRL.jl (Refactored)
# A Zero-Mutation, Type-Stable POSG Framework via Multiple Dispatch
# ===========================================================================
module GameRL

using StaticArrays
using Base.Threads # Import Julia's threading macros
using Statistics   # For calculating mean rewards

# ===========================================================================
# 1. API Exports
# The user can only access what you explicitly export.
# ===========================================================================
export AbstractGame, AbstractSimultaneousGame, AbstractSequentialGame
export legal_actions, transition, reward, observe, is_terminal, player_to_move
export ContinuousSpace, clip
export simulate_episode, run_episodes!, evaluate_parallel


struct WaitAction end

# ---------------------------------------------------------------------------
# The Interface API
# Every game must implement these 5 functions.
# ---------------------------------------------------------------------------
abstract type AbstractGame end

# The two major branches of Game Theory
abstract type AbstractSimultaneousGame <: AbstractGame end
abstract type AbstractSequentialGame <: AbstractGame end

# We define empty function signatures to document the required API
function legal_actions end
function transition end
function reward end
function observe end
function is_terminal end


# Required API for Sequential Games:
function player_to_move end  # (game, s) -> player_id
function legal_actions end   # (game, s, p) -> valid actions for active player `p`
function transition end      # (game, s, action) -> next_state
function reward end          # (game, s, action, s_next) -> NTuple of rewards for ALL players
function observe end         # (game, s, i) -> info_set
function is_terminal end     # (game, s) -> Bool

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

# ===========================================================================
# 2. Game Implementations (Concrete Types)
# ===========================================================================

"""
Multi-Armed Bandit (1 Player, 1 Step)
Strictly typed with an SVector to guarantee zero allocations.
"""
struct BanditGame{N} <: AbstractSimultaneousGame
    reward_probs::SVector{N, Float64}
end

# Implement the API for BanditGame
legal_actions(g::BanditGame{N}, s, i) where {N} = 1:N
transition(g::BanditGame, s, a_joint) = s + 1
reward(g::BanditGame, s, a_joint, s_next) = (rand() < g.reward_probs[a_joint[1]] ? 1.0 : 0.0,)
observe(g::BanditGame, s, i) = s
is_terminal(g::BanditGame, s) = s >= 1

"""
Normal Form / Matrix Game (N Players, Simultaneous, 1 Step)
"""
struct NormalFormGame{N, T} <: AbstractSimultaneousGame
    payoff_matrices::NTuple{N, T}
end

# Convenience constructor to allow varargs
NormalFormGame(matrices...) = NormalFormGame(matrices)

legal_actions(g::NormalFormGame, s, i) = 1:size(g.payoff_matrices[1], i)
transition(g::NormalFormGame, s, a_joint) = s + 1
reward(g::NormalFormGame, s, a_joint, s_next) = map(mat -> mat[a_joint...], g.payoff_matrices)
observe(g::NormalFormGame, s, i) = s
is_terminal(g::NormalFormGame, s) = s >= 1

"""
Population Normal Form Game
Strictly typed with a tuple of SMatrices.
"""
struct PopulationNormalFormGame{N, S} <: AbstractSimultaneousGame
    payoff_matrices::NTuple{N, S} # S will resolve to the specific SMatrix type
end

legal_actions(g::PopulationNormalFormGame, s, i) = size(g.payoff_matrices[i], i)
transition(g::PopulationNormalFormGame, s, a_joint) = s

function reward(g::PopulationNormalFormGame{N}, s, a_joint, s_next) where {N}
    # ntuple is evaluated at compile time if N is known, guaranteeing zero allocations
    return ntuple(N) do i
        opponent = i == 1 ? 2 : 1
        return g.payoff_matrices[i] * a_joint[opponent] 
    end
end

observe(g::PopulationNormalFormGame, s, i) = s
is_terminal(g::PopulationNormalFormGame, s) = false


"""
Bayesian Game (Incomplete Information)
State representation: `(stage, actual_types)`
"""
struct BayesianGame{S_fn, P_fn, A_tuple} <: AbstractSimultaneousGame
    sample_types_fn::S_fn
    payoff_fn::P_fn
    action_spaces::A_tuple
end

legal_actions(g::BayesianGame, s, i) = s[1] == 0 ? (1,) : g.action_spaces[i]
transition(g::BayesianGame, s, a_joint) = s[1] == 0 ? (1, g.sample_types_fn()) : (2, s[2])
reward(g::BayesianGame, s, a_joint, s_next) = s[1] == 1 ? g.payoff_fn(s[2], a_joint) : map(_ -> 0.0, g.action_spaces)
observe(g::BayesianGame, s, i) = s[1] == 0 ? nothing : s[2][i]
is_terminal(g::BayesianGame, s) = s[1] >= 2

"""
N-Player Polymatrix Population Game.
`payoff_matrices` is an NTuple of NTuples of SMatrices.
"""
struct PolymatrixGame{N, M} <: AbstractSimultaneousGame
    payoff_matrices::NTuple{N, M}
end

legal_actions(g::PolymatrixGame, s, i) = size(g.payoff_matrices[i][1], 1)
transition(g::PolymatrixGame, s, a_joint) = s

function reward(g::PolymatrixGame{N}, s, a_joint, s_next) where {N}
    # ntuple guarantees zero-allocation mapping at compile time
    return ntuple(N) do i
        interaction_vectors = ntuple(N) do j
            i == j ? zero(a_joint[i]) : g.payoff_matrices[i][j] * a_joint[j]
        end
        return sum(interaction_vectors)
    end
end

observe(g::PolymatrixGame, s, i) = s
is_terminal(g::PolymatrixGame, s) = false

"""
N-Player Mean Field Population Game.
`matrices` is an NTuple of SMatrices.
"""
struct MeanFieldGame{N, M} <: AbstractSimultaneousGame
    matrices::NTuple{N, M}
end

legal_actions(g::MeanFieldGame, s, i) = size(g.matrices[i], 1)
transition(g::MeanFieldGame, s, a_joint) = s

function reward(g::MeanFieldGame{N}, s, a_joint, s_next) where {N}
    return ntuple(N) do i
        others_actions = ntuple(N) do j
            i == j ? zero(a_joint[i]) : a_joint[j]
        end
        mean_field = sum(others_actions) ./ (N - 1)
        return g.matrices[i] * mean_field
    end
end

observe(g::MeanFieldGame, s, i) = s
is_terminal(g::MeanFieldGame, s) = false

# ===========================================================================
# 3. The Agent Protocol (Unchanged - Already Excellent)
# ===========================================================================

# ===========================================================================
# Discrete Agents
# ===========================================================================
struct QLearningAgent{N}
    q_values::SVector{N, Float64}
    counts::SVector{N, Int}
    epsilon::Float64
    alpha::Float64
end

function act(agent::QLearningAgent, obs, valid_actions::AbstractVector{<:Integer})
    if rand() < agent.epsilon
        return rand(valid_actions)
    else
        return argmax(agent.q_values)
    end
end

function learn(agent::QLearningAgent, obs, action::Integer, reward, next_obs)
    old_q = agent.q_values[action]
    new_q_val = old_q + agent.alpha * (reward - old_q)
    new_count_val = agent.counts[action] + 1
    
    # Using StaticArrays explicitly to ensure type-stable SVector updates
    new_qs = StaticArrays.setindex(agent.q_values, new_q_val, action)
    new_counts = StaticArrays.setindex(agent.counts, new_count_val, action)
    
    return QLearningAgent(new_qs, new_counts, agent.epsilon, agent.alpha)
end

# ===========================================================================
# Continuous Agents
# ===========================================================================

"""
A baseline agent that simply samples valid actions from a ContinuousSpace.
Zero allocations, completely type-stable.
"""
struct ContinuousRandomAgent end

# Notice the signature: It explicitly requires your ContinuousSpace struct!
function act(agent::ContinuousRandomAgent, obs, valid_actions::ContinuousSpace)
    return rand(valid_actions)
end

# It doesn't learn, but it must fulfill the interface and return itself.
function learn(agent::ContinuousRandomAgent, obs, action::SVector, reward, next_obs)
    return agent 
end


"""
A simple Continuous Actor (Linear Gaussian)
Maps an observation vector to an action vector continuously.
"""
struct LinearGaussianAgent{N_obs, N_act}
    weights::SMatrix{N_act, N_obs, Float64}
    std_dev::Float64
end

function act(agent::LinearGaussianAgent{N_obs, N_act}, obs::SVector{N_obs, Float64}, space::ContinuousSpace{N_act, Float64}) where {N_obs, N_act}
    # Deterministic mean action based on observation
    mean_action = agent.weights * obs
    
    # Add Gaussian noise for exploration (zero allocation using SVector)
    noise = agent.std_dev .* randn(SVector{N_act, Float64})
    
    # Clip the action to the valid continuous space bounds
    return clip(mean_action .+ noise, space)
end

function learn(agent::LinearGaussianAgent, obs, action::SVector, reward, next_obs)
    # Placeholder: In a real scenario, you'd do a policy gradient update here.
    # For now, we return the unmodified agent.
    return agent
end

# ===========================================================================
# 4. The Universal Simulation Engine
# ===========================================================================

"""
Simulates an episode. Notice we pass `game::AbstractSimultaneousGame`.
The compiler will devirtualize and inline the specific game logic dynamically!
"""
function simulate_episode(game::AbstractSimultaneousGame, initial_state, agents::Tuple; max_steps=100)
    s = initial_state
    current_agents = agents
    player_ids = Tuple(keys(agents)) 
    
    cumulative_rewards = map(_ -> 0.0, player_ids) 
    steps = 0
    
    # We now call the dispatch functions instead of struct fields
    while !is_terminal(game, s) && steps < max_steps
        
        obs = map(i -> observe(game, s, i), player_ids)
        valid_actions = map(i -> legal_actions(game, s, i), player_ids)
        
        a_joint = map(act, current_agents, obs, valid_actions)
        
        s_next = transition(game, s, a_joint)
        step_rewards = reward(game, s, a_joint, s_next)
        next_obs = map(i -> observe(game, s_next, i), player_ids)
        
        current_agents = map(learn, current_agents, obs, a_joint, step_rewards, next_obs)
        cumulative_rewards = map(+, cumulative_rewards, step_rewards)
        
        s = s_next
        steps += 1
    end
    
    return cumulative_rewards, steps, current_agents
end


# ===========================================================================
# Helper: Type-Stable Recursive Turn Engine
# ===========================================================================

# Base case: We ran out of agents (catches out-of-bounds errors)
_play_turn(game, s, p, current_idx, agents::Tuple{}) = error("Player $p out of bounds.")

# Recursive case: Check the first agent, then recurse on the rest
function _play_turn(game, s, p, current_idx, agents::Tuple)
    agent = first(agents)
    rest = Base.tail(agents)
    
    if p == current_idx
        # 1. It is this player's turn. Execute logic.
        obs = observe(game, s, p)
        valid_actions = legal_actions(game, s, p)
        action = act(agent, obs, valid_actions)
        
        # 2. Step the environment
        s_next = transition(game, s, action)
        step_rewards = reward(game, s, action, s_next)
        
        # 3. Learn
        next_obs = observe(game, s_next, p)
        updated_agent = learn(agent, obs, action, step_rewards[p], next_obs)
        
        # 4. Rebuild the tuple: updated agent + the rest of the unmodified tuple
        return s_next, step_rewards, (updated_agent, rest...)
    else
        # 1. Not this player's turn. Recurse to the next player.
        s_next, step_rewards, updated_rest = _play_turn(game, s, p, current_idx + 1, rest)
        
        # 2. Rebuild the tuple: unmodified current agent + the updated rest
        return s_next, step_rewards, (agent, updated_rest...)
    end
end

function simulate_episode(game::AbstractSequentialGame, initial_state, agents::Tuple; max_steps=100)
    s = initial_state
    current_agents = agents
    player_ids = Tuple(keys(agents)) 
    
    cumulative_rewards = map(_ -> 0.0, player_ids) 
    steps = 0
    
    while !is_terminal(game, s) && steps < max_steps
        
        # p is a dynamic integer known only at runtime
        p = player_to_move(game, s)
        
        # Enter the type-stable function barrier
        # We pass `1` as the starting index
        s_next, step_rewards, current_agents = _play_turn(game, s, p, 1, current_agents)
        
        cumulative_rewards = map(+, cumulative_rewards, step_rewards)
        s = s_next
        steps += 1
    end
    
    return cumulative_rewards, steps, current_agents
end

# ===========================================================================
# 5. The Driver Layer
# ===========================================================================

struct GameRunner{G <: AbstractGame, A <: Tuple}
    game::G
    initial_agents::A
end

function run_episodes!(runner::GameRunner, initial_env_state; n_episodes=100, log_every=100)
    player_ids = Tuple(keys(runner.initial_agents))
    total_rewards = map(_ -> 0.0, player_ids)
    
    current_agents = runner.initial_agents
    
    for ep in 1:n_episodes
        # Notice that current_agents is updated and passed back in, preserving learning!
        rewards, steps, current_agents = simulate_episode(
            runner.game, 
            initial_env_state, 
            current_agents
        )
        
        total_rewards = map(+, total_rewards, rewards)
        
        if ep % log_every == 0
            println("Episode $ep | Reward: ", round.(rewards, digits=2))
        end
    end
    
    return total_rewards, current_agents
end


# ===========================================================================
# Multi-Threaded Evaluation Engine
# ===========================================================================

"""
Runs `n_episodes` in parallel across all available CPU cores.
Perfectly thread-safe because your framework is zero-mutation!
"""
function evaluate_parallel(game::AbstractGame, initial_state, frozen_agents::Tuple; n_episodes=1000, max_steps=100)
    num_players = length(frozen_agents)
    
    # We allocate a thread-safe array to hold the total rewards for each episode
    # Rows = Episodes, Columns = Players
    results = zeros(Float64, n_episodes, num_players)
    
    # The magical Julia threading macro. It distributes the loop across your CPU cores.
    Threads.@threads for ep in 1:n_episodes
        # Because `simulate_episode` allocates everything on the stack and never
        # mutates `frozen_agents`, it is 100% thread-safe out of the box.
        rewards, steps, _ = simulate_episode(game, initial_state, frozen_agents, max_steps=max_steps)
        
        # Store the rewards safely
        for p in 1:num_players
            results[ep, p] = rewards[p]
        end
    end
    
    # Return the average reward per player across all parallel episodes
    avg_rewards = mean(results, dims=1)
    return avg_rewards
end

end # End of module GameRL



println("--- Booting RL Engine ---")

# FIX: You must use the @SVector macro here so the type matches BanditGame{3}
bandit = BanditGame(@SVector [0.1, 0.5, 0.9])

initial_q = @SVector [0.0, 0.0, 0.0]
initial_counts = @SVector [0, 0, 0]
agent = QLearningAgent(initial_q, initial_counts, 0.1, 0.1)

runner = GameRunner(bandit, (agent,))

println("Training for 5000 episodes...")
initial_state = 0 
total_rewards, final_agents = run_episodes!(runner, initial_state, n_episodes=5000, log_every=1000)

println("\n--- Results ---")
println("Final Q-Values: ", round.(final_agents[1].q_values, digits=3))
println("Arm Pull Counts: ", final_agents[1].counts)