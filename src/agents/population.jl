# ===========================================================================
# src/agents/population.jl
# ===========================================================================

"""
Replicator Dynamics Agent
Models a continuous population distribution over `N` pure strategies.
Instead of picking a single action, it outputs the probability distribution
(the population fractions) directly.
"""
struct ReplicatorAgent{N}
    population_fractions::SVector{N, Float64}
    step_size::Float64
end

# Added the `game` argument to comply with the new API
function act(agent::ReplicatorAgent, game, obs, valid_actions)
    # The action *is* the continuous probability distribution
    return agent.population_fractions
end

# Added the `game` argument
function learn(agent::ReplicatorAgent, game, obs, current_dist, expected_payoffs, next_obs)
    # 1. Calculate the average population payoff: x^T * u
    avg_u = sum(current_dist .* expected_payoffs)
    
    # 2. Replicator Equation: dx/dt = x_i * (u_i - avg_u)
    # Discretized via simple Euler method: x_new = x_old + step_size * dx/dt
    deltas = agent.step_size .* current_dist .* (expected_payoffs .- avg_u)
    new_fractions_raw = current_dist .+ deltas
    
    # 3. Project back to the simplex (handle floating point drift)
    # Ensures the population fractions always sum exactly to 1.0
    new_fractions = new_fractions_raw ./ sum(new_fractions_raw)
    
    # 4. Zero-mutation return
    return ReplicatorAgent(new_fractions, agent.step_size)
end