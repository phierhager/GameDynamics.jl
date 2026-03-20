# ===========================================================================
# GameRL.jl
# A Zero-Mutation, Type-Stable POSG Framework for RL and Game Theory
# ===========================================================================

using StaticArrays

struct WaitAction end

# ===========================================================================
# 1. The Core Kernel
# ===========================================================================

"""
The universal Partially Observable Stochastic Game (POSG).
Parameterized to guarantee C-like performance without type instability.
"""
struct FunctionalGame{A, T, R, O, Z}
    legal_actions::A   # (state, player_id) -> Iterable of actions
    transition::T      # (state, joint_action) -> next_state
    reward::R          # (state, joint_action, next_state) -> Tuple of rewards
    observe::O         # (state, player_id) -> observation / info_set
    is_terminal::Z     # (state) -> Bool
end

# ===========================================================================
# 2. Game Constructors (Type-Stable via `let` blocks)
# ===========================================================================

"""
Multi-Armed Bandit (1 Player, 1 Step)
Strictly typed. Requires `reward_probs` to be an SVector.
"""
function make_bandit(reward_probs::SVector{N, Float64}) where {N}
    # No let block needed for `reward_probs` because its type and length 
    # are fully resolved by the `where {N}` dispatch!
    return FunctionalGame(
        (s, i) -> 1:N,
        (s, a) -> s + 1,                                  
        (s, a, s_next) -> (rand() < reward_probs[a[1]] ? 1.0 : 0.0,), 
        (s, i) -> s,                                      
        s -> s >= 1                                       
    )
end

# Usage:
# bandit = make_bandit(@SVector [0.1, 0.5, 0.9])

"""
Normal Form / Matrix Game (N Players, Simultaneous, 1 Step)
"""
function make_normal_form(payoff_matrices...)
    let matrices = payoff_matrices
        return FunctionalGame(
            (s, i) -> 1:size(matrices[1], i),
            (s, a) -> s + 1,                                  
            (s, a, s_next) -> map(mat -> mat[a...], matrices),
            (s, i) -> s,
            s -> s >= 1                                       
        )
    end
end

"""
Population Normal Form Game
Strictly typed to SMatrix to guarantee zero-allocation matrix multiplication.
"""
function make_population_normal_form(payoff_matrices::SMatrix...)
    num_players = length(payoff_matrices)
    
    let matrices = payoff_matrices, N = num_players
        return FunctionalGame(
            (s, i) -> size(matrices[i], i),
            (s, a_joint) -> s,
            (s, a_joint, s_next) -> ntuple(N) do i
                opponent = i == 1 ? 2 : 1
                # Because matrices[i] is an SMatrix and a_joint[opponent] 
                # is an SVector, this multiplication is stack-allocated!
                return matrices[i] * a_joint[opponent] 
            end,
            (s, i) -> s,
            s -> false 
        )
    end
end

"""
Markov Decision Process (1 Player, Stateful)
"""
function make_mdp(actions_fn, transition_fn, reward_fn, terminal_fn)
    return FunctionalGame(
        (s, i) -> actions_fn(s),
        (s, a) -> transition_fn(s, a[1]),
        (s, a, s_next) -> (reward_fn(s, a[1], s_next),),
        (s, i) -> s,                                          # Perfect Info
        s -> terminal_fn(s)
    )
end

"""
Markov Game (N Players, Stateful, Simultaneous)
"""
function make_markov_game(actions_fn, transition_fn, reward_fn, terminal_fn)
    return FunctionalGame(
        (s, i) -> actions_fn(s, i),
        (s, a) -> transition_fn(s, a),
        (s, a, s_next) -> reward_fn(s, a, s_next),
        (s, i) -> s,
        s -> terminal_fn(s)
    )
end

"""
Partially Observable Markov Decision Process (POMDP)
1 Player. The key difference from an MDP is that `observe` applies an 
observation function rather than returning the raw state.
"""
function make_pomdp(actions_fn, transition_fn, reward_fn, observation_fn, terminal_fn)
    return FunctionalGame(
        (s, i) -> actions_fn(s),
        (s, a) -> transition_fn(s, a[1]),
        (s, a, s_next) -> (reward_fn(s, a[1], s_next),),
        (s, i) -> observation_fn(s),                          # Partial Info via O(s)
        s -> terminal_fn(s)
    )
end

"""
Bayesian Game (Incomplete Information)
Nature draws types in step 0. Players act in step 1.
State representation: `(stage, actual_types)`
"""
function make_bayesian(sample_types_fn, payoff_fn, action_spaces::Tuple)
    let spaces = action_spaces
        return FunctionalGame(
            (s, i) -> s[1] == 0 ? (1,) : spaces[i],
            (s, a) -> s[1] == 0 ? (1, sample_types_fn()) : (2, s[2]),
            # CHANGED: Map over the action_spaces to generate a zeros tuple of the correct size
            (s, a, s_next) -> s[1] == 1 ? payoff_fn(s[2], a) : map(_ -> 0.0, spaces),
            (s, i) -> s[1] == 0 ? nothing : s[2][i],
            s -> s[1] >= 2                                    
        )
    end
end

"""
Extensive Form Game (Sequential Play)
Players take turns based on a `player_to_move` function.
Inactive players are forced to output a dummy action (e.g., 0) to maintain the `a_joint` tuple.
"""
function make_extensive_form(player_to_move_fn, legal_actions_fn, transition_fn, reward_fn, info_set_fn, is_terminal_fn)
    # Using let to lock down function boundaries
    let p_move = player_to_move_fn
        return FunctionalGame(
            # If it's your turn, return legal actions. Otherwise, return dummy WaitAction().
            (s, i) -> p_move(s) == i ? legal_actions_fn(s) : WaitAction(),
            
            # Transition only processes the action of the active player
            (s, a) -> transition_fn(s, a[p_move(s)]),
            
            # Rewards might be emitted sequentially or only at terminal states
            (s, a, s_next) -> reward_fn(s, a[p_move(s)], s_next),
            
            # Maps the game node `s` to an information set for player `i`
            (s, i) -> info_set_fn(s, i),
            
            s -> is_terminal_fn(s)
        )
    end
end


"""
N-Player Polymatrix Population Game.
`payoff_matrices` should be a Tuple of Tuples of SMatrices.
e.g., matrices[i][j] is the payoff matrix for player i playing against player j.
"""
function make_n_player_polymatrix(payoff_matrices::Tuple)
    let matrices = payoff_matrices
        player_ids = keys(matrices) # Generates a strict tuple: (1, 2, ..., N)
        
        return FunctionalGame(
            (s, i) -> size(matrices[i][1], 1), # Assumes square/compatible matrices
            (s, a_joint) -> s,
            
            (s, a_joint, s_next) -> map(player_ids) do i
                
                # 1. Map over all opponents to generate a TUPLE of payoff vectors
                interaction_vectors = map(player_ids) do j
                    if i == j
                        # Return a zero vector of the exact same static type/size
                        zero(a_joint[i]) 
                    else
                        # Matrix * Vector -> SVector
                        matrices[i][j] * a_joint[j] 
                    end
                end
                
                # 2. Sum the tuple of SVectors. 
                # Because the length is known at compile time, Julia unrolls this 
                # entirely into `v1 + v2 + ... + vN` without a single allocation.
                return sum(interaction_vectors)
            end,
            
            (s, i) -> s,
            s -> false 
        )
    end
end



"""
N-Player Mean Field Population Game.
`matrices` is a Tuple of SMatrices (one for each player).
"""
function make_mean_field_population(matrices::Tuple)
    let N = length(matrices), mats = matrices
        player_ids = keys(mats)
        
        return FunctionalGame(
            (s, i) -> size(mats[i], 1),
            (s, a_joint) -> s,
            
            (s, a_joint, s_next) -> map(player_ids) do i
                
                # 1. Gather the actions of everyone EXCEPT i
                others_actions = map(player_ids) do j
                    i == j ? zero(a_joint[i]) : a_joint[j]
                end
                
                # 2. Compute the mean field
                mean_field = sum(others_actions) ./ (N - 1)
                
                # 3. Single matrix multiplication
                return mats[i] * mean_field
            end,
            
            (s, i) -> s,
            s -> false 
        )
    end
end

# ===========================================================================
# 3. The Agent Protocol (Causal, Zero-Mutation)
# ===========================================================================

# Any valid agent in this framework MUST implement `act` and `learn`.

"""
Stateful Q-Learning Agent using StaticArrays for zero-allocation updates.
`N` is the number of available actions.
"""
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
    # 1. Calculate new values
    old_q = agent.q_values[action]
    new_q_val = old_q + agent.alpha * (reward - old_q)
    new_count_val = agent.counts[action] + 1
    
    # 2. Functional update via StaticArrays (Creates a new SVector on the stack)
    new_qs = setindex(agent.q_values, new_q_val, action)
    new_counts = setindex(agent.counts, new_count_val, action)
    
    # 3. Return the exact same struct type with updated data
    return QLearningAgent(new_qs, new_counts, agent.epsilon, agent.alpha)
end

"""
Replicator Dynamics Agent
Models a continuous population distribution over `N` pure strategies.
"""
struct ReplicatorAgent{N}
    population_fractions::SVector{N, Float64}
    step_size::Float64
end

function act(agent::ReplicatorAgent, obs, valid_actions)
    # Instead of an integer, the agent outputs the continuous probability distribution
    return agent.population_fractions
end

function learn(agent::ReplicatorAgent, obs, current_dist, expected_payoffs, next_obs)
    # 1. Calculate the average population payoff: x^T * u
    avg_u = sum(current_dist .* expected_payoffs)
    
    # 2. Replicator Equation: dx/dt = x_i * (u_i - avg_u)
    # Discretized via Euler method: x_new = x_old + alpha * dx/dt
    deltas = agent.step_size .* current_dist .* (expected_payoffs .- avg_u)
    new_fractions_raw = current_dist .+ deltas
    
    # 3. Project back to the simplex (handle floating point drift)
    new_fractions = new_fractions_raw ./ sum(new_fractions_raw)
    
    # 4. Zero-mutation return
    return ReplicatorAgent(new_fractions, agent.step_size)
end

# ===========================================================================
# The Universal Simulation Engine
# ===========================================================================

"""
Simulates an episode for ANY agent type (Discrete or Continuous).
Julia's compiler will specialize this loop based on the return types of `act` and `reward`.
"""
function simulate_episode(game::FunctionalGame, initial_state, agents::Tuple; max_steps=100)
    s = initial_state
    current_agents = agents
    
    # CHANGED: keys(agents) creates a type-stable tuple of indices: (1, 2, ..., N)
    player_ids = keys(agents) 
    
    # Initialize accumulator as a type-stable tuple of zeros
    cumulative_rewards = map(_ -> 0.0, player_ids) 
    
    steps = 0
    
    while !game.is_terminal(s) && steps < max_steps
        
        # 1. Map over the player IDs to get observations and valid actions
        obs = map(i -> game.observe(s, i), player_ids)
        valid_actions = map(i -> game.legal_actions(s, i), player_ids)
        
        # 2. Map across tuples to get joint actions (Zero dynamic dispatch)
        a_joint = map(act, current_agents, obs, valid_actions)
        
        # 3. Environment transition
        s_next = game.transition(s, a_joint)
        step_rewards = game.reward(s, a_joint, s_next)
        next_obs = map(i -> game.observe(s_next, i), player_ids)
        
        # 4. Map the learn function across all heterogeneous agents simultaneously
        current_agents = map(learn, current_agents, obs, a_joint, step_rewards, next_obs)
        
        # 5. Type-stable tuple addition for rewards
        cumulative_rewards = map(+, cumulative_rewards, step_rewards)
        
        s = s_next
        steps += 1
    end
    
    return cumulative_rewards, steps, current_agents
end

# ===========================================================================
# 5. The Driver Layer
# ===========================================================================

struct GameRunner{G, A}
    game::G
    initial_agents::A
end

"""
Executes the training loop over multiple episodes.
"""
function run_episodes!(runner::GameRunner, initial_env_state; n_episodes=100, log_every=100)
    player_ids = keys(runner.initial_agents)
    
    # CHANGED: Replaced zeros() heap allocation with a purely functional tuple
    total_rewards = map(_ -> 0.0, player_ids)
    
    current_agents = runner.initial_agents
    
    for ep in 1:n_episodes
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
# 6. Usage Example: Solving the Bandit
# ===========================================================================

println("--- Booting RL Engine ---")

# 1. Define the game (3 arms: 10%, 50%, 90% win rates)
bandit = make_bandit(@SVector [0.1, 0.5, 0.9])

# 2. Initialize the Q-Learning Agent
# SVector guarantees we don't trigger the garbage collector in the hot loop
initial_q = @SVector [0.0, 0.0, 0.0]
initial_counts = @SVector [0, 0, 0]
agent = QLearningAgent(initial_q, initial_counts, 0.1, 0.1)

# 3. Setup the Runner
runner = GameRunner(bandit, (agent,))

# 4. Train
println("Training for 5000 episodes...")
initial_state = 0 # Stateless games start at 0, transition to 1, and terminate
total_rewards, final_agents = run_episodes!(runner, initial_state, n_episodes=5000, log_every=1000)

# 5. Results
println("\n--- Results ---")
println("Final Q-Values: ", round.(final_agents[1].q_values, digits=3))
println("Arm Pull Counts: ", final_agents[1].counts)
println("The agent should have converged on Arm 3 as the optimal choice.")