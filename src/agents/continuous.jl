# ===========================================================================
# src/agents/continuous.jl
# ===========================================================================

struct ContinuousRandomAgent end

function act(agent::ContinuousRandomAgent, game, obs, valid_actions::ContinuousSpace)
    return rand(valid_actions)
end

function learn(agent::ContinuousRandomAgent, game, obs, action::SVector, reward, next_obs)
    return agent 
end


struct LinearGaussianAgent{N_obs, N_act}
    weights::SMatrix{N_act, N_obs, Float64}
    std_dev::Float64
end

function act(agent::LinearGaussianAgent{N_obs, N_act}, game, obs::SVector{N_obs, Float64}, space::ContinuousSpace{N_act, Float64}) where {N_obs, N_act}
    mean_action = agent.weights * obs
    noise = agent.std_dev .* randn(SVector{N_act, Float64})
    return clip(mean_action .+ noise, space)
end

function learn(agent::LinearGaussianAgent, game, obs, action::SVector, reward, next_obs)
    return agent
end