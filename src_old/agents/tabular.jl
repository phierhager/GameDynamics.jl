# ===========================================================================
# src/agents/tabular.jl
# ===========================================================================

struct QLearningAgent{N}
    q_values::SVector{N, Float64}
    counts::SVector{N, Int}
    epsilon::Float64
    alpha::Float64
end

function act(agent::QLearningAgent, game, obs, valid_actions::AbstractVector{<:Integer})
    if rand() < agent.epsilon
        return rand(valid_actions)
    else
        return argmax(agent.q_values)
    end
end

function learn(agent::QLearningAgent, game, obs, action::Integer, reward, next_obs)
    old_q = agent.q_values[action]
    new_q_val = old_q + agent.alpha * (reward - old_q)
    new_count_val = agent.counts[action] + 1
    
    new_qs = StaticArrays.setindex(agent.q_values, new_q_val, action)
    new_counts = StaticArrays.setindex(agent.counts, new_count_val, action)
    
    return QLearningAgent(new_qs, new_counts, agent.epsilon, agent.alpha)
end