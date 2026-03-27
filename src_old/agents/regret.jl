# ===========================================================================
# src/agents/regret.jl
# ===========================================================================
using StaticArrays
import ..Core: counterfactual_rewards

"""
Regret Matching Agent
Converges to a Coarse Correlated Equilibrium (CCE) in general-sum games, 
and Nash Equilibrium (NE) in zero-sum games.
"""
struct RegretMatchingAgent{N}
    cumulative_regrets::SVector{N, Float64}
    strategy_sum::SVector{N, Float64} # The average strategy is the actual equilibrium!
    player_id::Int # Needed to extract the right counterfactuals
end

function act(agent::RegretMatchingAgent{N}, game, obs, valid_actions) where {N}
    positive_regrets = max.(agent.cumulative_regrets, 0.0)
    sum_regret = sum(positive_regrets)
    
    # If there's positive regret, play proportionally. Otherwise, play uniformly.
    strategy = sum_regret > 0.0 ? (positive_regrets ./ sum_regret) : fill(1.0/N, SVector{N, Float64})
    
    # Zero-allocation categorical sampling
    r = rand()
    cum_p = 0.0
    for i in 1:N
        cum_p += strategy[i]
        if r <= cum_p
            return valid_actions[i]
        end
    end
    return valid_actions[N]
end

function learn(agent::RegretMatchingAgent{N}, game, obs, a_joint, reward, next_obs) where {N}
    # 1. Calculate the strategy we just played (to update the average strategy)
    positive_regrets = max.(agent.cumulative_regrets, 0.0)
    sum_regret = sum(positive_regrets)
    current_strategy = sum_regret > 0.0 ? (positive_regrets ./ sum_regret) : fill(1.0/N, SVector{N, Float64})
    
    # 2. Get counterfactual rewards (requires a new API function on your games)
    # This returns an SVector of what the agent WOULD have gotten for every possible action
    cf_rewards = counterfactual_rewards(game, agent.player_id, a_joint)
    
    # 3. Calculate regrets: (CF reward for action A) - (Actual reward received)
    instant_regrets = cf_rewards .- reward
    
    # 4. Zero-mutation return
    new_regrets = agent.cumulative_regrets .+ instant_regrets
    new_strategy_sum = agent.strategy_sum .+ current_strategy
    
    return RegretMatchingAgent{N}(new_regrets, new_strategy_sum, agent.player_id)
end


"""
Fictitious Play Agent (2-Player)
Converges to Nash Equilibrium in zero-sum and potential games.
"""
struct FictitiousPlayAgent{N_actions, N_opp_actions}
    opp_action_counts::SVector{N_opp_actions, Int}
    player_id::Int
end

function act(agent::FictitiousPlayAgent{N, M}, game, obs, valid_actions) where {N, M}
    # 1. Estimate opponent's strategy
    total_opp_actions = sum(agent.opp_action_counts)
    if total_opp_actions == 0
        return rand(valid_actions) # Play randomly on turn 1
    end
    opp_strategy = agent.opp_action_counts ./ total_opp_actions
    
    # 2. Calculate Expected Utility for all our actions against opp_strategy
    # We grab our payoff matrix from the game object
    my_matrix = game.payoff_matrices[agent.player_id]
    
    # expected_utilities = Matrix * opponent_strategy_vector
    # If player 1, matrix is (N x M). If player 2, matrix is (M x N) and we need to transpose it conceptually.
    expected_utilities = agent.player_id == 1 ? (my_matrix * opp_strategy) : (opp_strategy' * my_matrix)'
    
    # 3. Play the Best Response (argmax)
    best_action_idx = argmax(expected_utilities)
    return valid_actions[best_action_idx]
end

function learn(agent::FictitiousPlayAgent{N, M}, game, obs, a_joint, reward, next_obs) where {N, M}
    # Identify what the opponent did
    opp_id = agent.player_id == 1 ? 2 : 1
    opp_action = a_joint[opp_id]
    
    # Increment the count for the opponent's action
    new_counts = Base.setindex(agent.opp_action_counts, agent.opp_action_counts[opp_action] + 1, opp_action)
    
    return FictitiousPlayAgent{N, M}(new_counts, agent.player_id)
end


"""
Internal Regret Agent
Converges to the Correlated Equilibrium (CE) set.
"""
struct InternalRegretAgent{N}
    regret_matrix::SMatrix{N, N, Float64} # regret[i, j] is regret for playing i instead of j
    strategy_sum::SVector{N, Float64}
    player_id::Int
end

function act(agent::InternalRegretAgent{N}, game, obs, valid_actions) where {N}
    # To find the next strategy, we find the stationary distribution 
    # of the transition matrix formed by positive internal regrets.
    # For a 2x2 or 3x3, we can approximate or use the standard RM-swap logic.
    
    # Simplified version: Play proportionally to the sum of incoming regrets
    incoming_regrets = vec(sum(max.(agent.regret_matrix, 0.0), dims=1))
    sum_reg = sum(incoming_regrets)
    strategy = sum_reg > 0.0 ? incoming_regrets ./ sum_reg : fill(1.0/N, SVector{N, Float64})
    
    # Standard zero-allocation sampling
    r = rand(); cum_p = 0.0
    for i in 1:N
        cum_p += strategy[i]
        if r <= cum_p; return valid_actions[i]; end
    end
    return valid_actions[N]
end

function learn(agent::InternalRegretAgent{N}, game, obs, a_joint, reward, next_obs) where {N}
    my_action = a_joint[agent.player_id]
    cf_rewards = counterfactual_rewards(game, agent.player_id, a_joint)
    
    # Calculate the new regret row for the action we just took
    new_regrets_for_action = cf_rewards .- reward
    
    # Reconstruct the matrix row-by-row
    # We use ntuple to get each row, then vcat them into a single flat vector, 
    # then reshape into the SMatrix.
    rows = ntuple(i -> i == my_action ? 
        (agent.regret_matrix[i, :] .+ new_regrets_for_action) : 
        agent.regret_matrix[i, :], N)
    
    # This vcat(rows...) trick ensures StaticArrays gets a flat stream of numbers
    new_matrix = SMatrix{N, N, Float64}(vcat(rows...))
    
    # Update strategy sum (one-hot vector for the action taken)
    current_strat = SVector{N, Float64}(ntuple(i -> i == my_action ? 1.0 : 0.0, N))
    
    return InternalRegretAgent{N}(new_matrix, agent.strategy_sum .+ current_strat, agent.player_id)
end