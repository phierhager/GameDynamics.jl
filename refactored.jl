# ===========================================================================
# GameRL.jl (Refactored)
# A Zero-Mutation, Type-Stable POSG Framework via Multiple Dispatch
# ===========================================================================

using StaticArrays

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
Markov Decision Process (1 Player, Stateful)
"""
struct MDP{A, T, R, Z} <: AbstractSimultaneousGame
    actions_fn::A
    transition_fn::T
    reward_fn::R
    terminal_fn::Z
end

legal_actions(g::MDP, s, i) = g.actions_fn(s)
transition(g::MDP, s, a_joint) = g.transition_fn(s, a_joint[1])
reward(g::MDP, s, a_joint, s_next) = (g.reward_fn(s, a_joint[1], s_next),)
observe(g::MDP, s, i) = s # Perfect Info
is_terminal(g::MDP, s) = g.terminal_fn(s)

"""
Markov Game (N Players, Stateful, Simultaneous)
"""
struct MarkovGame{A, T, R, Z} <: AbstractSimultaneousGame
    actions_fn::A
    transition_fn::T
    reward_fn::R
    terminal_fn::Z
end

legal_actions(g::MarkovGame, s, i) = g.actions_fn(s, i)
transition(g::MarkovGame, s, a_joint) = g.transition_fn(s, a_joint)
reward(g::MarkovGame, s, a_joint, s_next) = g.reward_fn(s, a_joint, s_next)
observe(g::MarkovGame, s, i) = s
is_terminal(g::MarkovGame, s) = g.terminal_fn(s)

"""
Partially Observable Markov Decision Process (POMDP)
"""
struct POMDP{A, T, R, O, Z} <: AbstractSimultaneousGame
    actions_fn::A
    transition_fn::T
    reward_fn::R
    observation_fn::O
    terminal_fn::Z
end

legal_actions(g::POMDP, s, i) = g.actions_fn(s)
transition(g::POMDP, s, a_joint) = g.transition_fn(s, a_joint[1])
reward(g::POMDP, s, a_joint, s_next) = (g.reward_fn(s, a_joint[1], s_next),)
observe(g::POMDP, s, i) = g.observation_fn(s) # Partial Info via O(s)
is_terminal(g::POMDP, s) = g.terminal_fn(s)

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


"""
Extensive Form Game (Sequential Play)
Parameterized to maintain type stability.
"""
struct ExtensiveFormGame{P, A, T, R, I, Z} <: AbstractSequentialGame
    player_to_move_fn::P
    legal_actions_fn::A
    transition_fn::T
    reward_fn::R
    info_set_fn::I
    is_terminal_fn::Z
end

# Implement the API
player_to_move(g::ExtensiveFormGame, s) = g.player_to_move_fn(s)
legal_actions(g::ExtensiveFormGame, s, p) = g.legal_actions_fn(s, p)
transition(g::ExtensiveFormGame, s, action) = g.transition_fn(s, action)
reward(g::ExtensiveFormGame, s, action, s_next) = g.reward_fn(s, action, s_next)
observe(g::ExtensiveFormGame, s, i) = g.info_set_fn(s, i)
is_terminal(g::ExtensiveFormGame, s) = g.is_terminal_fn(s)

# ===========================================================================
# 3. The Agent Protocol (Unchanged - Already Excellent)
# ===========================================================================

struct QLearningAgent{N}
    q_values::SVector{N, Float64}
    counts::SVector{N, Int}
    epsilon::Float64
    alpha::Float64
end

function act(agent::QLearningAgent, obs, valid_actions)
    if rand() < agent.epsilon
        return rand(valid_actions)
    else
        return argmax(agent.q_values)
    end
end

function learn(agent::QLearningAgent, obs, action, reward, next_obs)
    old_q = agent.q_values[action]
    new_q_val = old_q + agent.alpha * (reward - old_q)
    new_count_val = agent.counts[action] + 1
    
    new_qs = setindex(agent.q_values, new_q_val, action)
    new_counts = setindex(agent.counts, new_count_val, action)
    
    return QLearningAgent(new_qs, new_counts, agent.epsilon, agent.alpha)
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
    player_ids = keys(agents) 
    
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


"""
Simulates an episode for Sequential Games (Turn-Based).
Only the active player acts and learns. Zero allocations.
"""
function simulate_episode(game::AbstractSequentialGame, initial_state, agents::Tuple; max_steps=100)
    s = initial_state
    current_agents = agents
    player_ids = keys(agents) 
    
    cumulative_rewards = map(_ -> 0.0, player_ids) 
    steps = 0
    
    while !is_terminal(game, s) && steps < max_steps
        
        # 1. Identify whose turn it is
        p = player_to_move(game, s)
        
        # 2. Only the active player observes and acts
        obs = observe(game, s, p)
        valid_actions = legal_actions(game, s, p)
        action = act(current_agents[p], obs, valid_actions)
        
        # 3. Environment transitions based on a SINGLE action
        s_next = transition(game, s, action)
        
        # Reward is still an N-element tuple so everyone can get payoffs 
        # (e.g., Player 1 acts, resulting in a win for Player 1 and loss for Player 2)
        step_rewards = reward(game, s, action, s_next)
        
        # 4. Active player observes the result and learns
        next_obs = observe(game, s_next, p)
        updated_agent = learn(current_agents[p], obs, action, step_rewards[p], next_obs)
        
        # 5. THE JULIAN TRICK: Swap the updated agent back into the immutable tuple.
        # This returns a brand new tuple on the stack. Zero garbage collection!
        current_agents = Base.setindex(current_agents, updated_agent, p)
        
        # Accumulate rewards for everyone
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
    player_ids = keys(runner.initial_agents)
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