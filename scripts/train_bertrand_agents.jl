#!/usr/bin/env julia

# scripts/train_bertrand_agents.jl

include(joinpath(@__DIR__, "..", "src", "GameLab.jl"))
using .GameLab
using Random
using Printf
using Statistics

# ------------------------------------------------------------------------------
# Agent wrapper
# ------------------------------------------------------------------------------

mutable struct TrainableAgent
    name::String
    learner
    state
    mode::Symbol  # :bandit or :full_information
    action_counts::Vector{Int}
    reward_trace::Vector{Float64}
end

function TrainableAgent(name::String, learner, state, mode::Symbol, n_actions::Int)
    mode in (:bandit, :full_information) ||
        throw(ArgumentError("Unsupported agent mode: $mode"))

    return TrainableAgent(
        name,
        learner,
        state,
        mode,
        zeros(Int, n_actions),
        Float64[],
    )
end

# ------------------------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------------------------

function get_policy(agent::TrainableAgent, n_actions::Int; rng::AbstractRNG = Random.default_rng())
    π = zeros(Float64, n_actions)

    try
        GameLab.LearningInterfaces.policy!(π, agent.learner, agent.state, nothing)
        return π
    catch err
        if err isa MethodError || err isa ArgumentError
            a = GameLab.LearningInterfaces.act!(agent.learner, agent.state, nothing, rng)
            π .= 0.0
            π[a] = 1.0
            return π
        end
        rethrow()
    end
end

function choose_action!(agent::TrainableAgent, rng::AbstractRNG)
    a = GameLab.LearningInterfaces.act!(agent.learner, agent.state, nothing, rng)
    agent.action_counts[a] += 1
    return a
end

function reset_agent!(agent::TrainableAgent)
    GameLab.LearningInterfaces.reset!(agent.learner, agent.state)
    fill!(agent.action_counts, 0)
    empty!(agent.reward_trace)
    return agent
end

# ------------------------------------------------------------------------------
# Counterfactual utility vector for full-information learners
# ------------------------------------------------------------------------------

function full_information_feedback_homogeneous(game, joint_actions::NTuple{N,Int}, player::Int) where {N}
    n_actions = length(game.price_grid)
    utilities = zeros(Float64, n_actions)

    for a in 1:n_actions
        alt_profile = ntuple(i -> i == player ? a : joint_actions[i], N)
        s0 = GameLab.Kernel.init_state(game)
        _, reward = GameLab.Kernel.step(game, s0, GameLab.Kernel.JointAction(alt_profile))
        utilities[a] = reward[player]
    end

    return utilities
end

# ------------------------------------------------------------------------------
# Main repeated-play loop
# ------------------------------------------------------------------------------

function run_training!(
    game,
    agents::Vector{TrainableAgent};
    rounds::Int = 20_000,
    rng::AbstractRNG = MersenneTwister(42),
    log_every::Int = 1_000,
)
    N = length(agents)
    N == GameLab.Kernel.num_players(game) ||
        throw(ArgumentError("Agent count must match number of players."))

    n_actions = length(game.price_grid)
    mean_rewards = zeros(Float64, N)
    last_prices = zeros(Float64, N)

    for t in 1:rounds
        s0 = GameLab.Kernel.init_state(game, rng)

        actions = ntuple(i -> choose_action!(agents[i], rng), N)
        ja = GameLab.Kernel.JointAction(actions)

        _, reward = GameLab.Kernel.step(game, s0, ja, rng)

        @inbounds for i in 1:N
            last_prices[i] = game.price_grid[actions[i]]
            push!(agents[i].reward_trace, reward[i])
        end

        @inbounds for i in 1:N
            agent = agents[i]

            if agent.mode == :bandit
                rec = GameLab.RuntimeRecords.BanditRecord(actions[i], reward[i], true)
                GameLab.LearningInterfaces.update!(agent.learner, agent.state, rec)

            elseif agent.mode == :full_information
                feedback = full_information_feedback_homogeneous(game, actions, i)
                rec = GameLab.RuntimeRecords.FullInformationRecord(actions[i], feedback, true)
                GameLab.LearningInterfaces.update!(agent.learner, agent.state, rec)

            else
                error("Unknown agent mode $(agent.mode)")
            end
        end

        @inbounds for i in 1:N
            mean_rewards[i] += (reward[i] - mean_rewards[i]) / t
        end

        if t == 1 || t % log_every == 0 || t == rounds
            println("round = $t")
            for i in 1:N
                π = get_policy(agents[i], n_actions; rng = rng)
                best_idx = argmax(π)
                @printf(
                    "  player=%d  agent=%-14s  avg_reward=%8.4f  last_price=%8.3f  modal_price=%8.3f\n",
                    i,
                    agents[i].name,
                    mean_rewards[i],
                    last_prices[i],
                    game.price_grid[best_idx],
                )
            end
            println()
        end
    end

    return nothing
end

# ------------------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------------------

function summarize_agents(game, agents::Vector{TrainableAgent})
    n_actions = length(game.price_grid)

    println("=== final summary ===")
    for (i, agent) in enumerate(agents)
        avg_reward = isempty(agent.reward_trace) ? 0.0 : mean(agent.reward_trace)

        freq = zeros(Float64, n_actions)
        GameLab.AnalysisEvaluation.action_frequency_report!(freq, agent.action_counts)

        modal_action = argmax(freq)
        modal_price = game.price_grid[modal_action]
        π = get_policy(agent, n_actions)

        println("player $i / $(agent.name)")
        @printf("  average reward: %.6f\n", avg_reward)
        @printf("  modal action:   %d\n", modal_action)
        @printf("  modal price:    %.6f\n", modal_price)
        println("  empirical frequencies = ", round.(freq; digits = 4))
        println("  current policy        = ", round.(π; digits = 4))
        println()
    end
end

# ------------------------------------------------------------------------------
# Example game builders
# ------------------------------------------------------------------------------

function build_homogeneous_bertrand_game()
    a = 12.0
    b = 1.0

    price_grid = collect(0.0:0.5:10.0)
    marginal_costs = (2.0, 2.0)

    demand_curve(p) = max(0.0, a - b * p)

    return GameLab.HomogeneousBertrand.HomogeneousBertrandGame(
        price_grid,
        marginal_costs,
        demand_curve,
    )
end

function build_differentiated_bertrand_game()
    price_grid = collect(0.0:0.5:10.0)
    attractiveness = (4.0, 4.2)
    marginal_costs = (2.0, 2.0)
    price_sensitivity = 1.0
    outside_option_utility = 0.0

    demand_curve(prices, shares) = 10.0

    return GameLab.DifferentiatedBertrand.DifferentiatedBertrandGame(
        price_grid,
        attractiveness,
        price_sensitivity,
        outside_option_utility,
        marginal_costs,
        demand_curve,
    )
end

# ------------------------------------------------------------------------------
# Agent factory
# ------------------------------------------------------------------------------

function build_agents(n_actions::Int)
    agents = TrainableAgent[]

    exp3 = GameLab.EXP3Learners.EXP3(0.07, 0.07, n_actions)
    push!(agents, TrainableAgent(
        "EXP3",
        exp3,
        GameLab.EXP3Learners.EXP3State(exp3),
        :bandit,
        n_actions,
    ))

    hedge = GameLab.HedgeLearners.Hedge(0.08, n_actions)
    push!(agents, TrainableAgent(
        "Hedge",
        hedge,
        GameLab.HedgeLearners.HedgeState(hedge),
        :full_information,
        n_actions,
    ))

    return agents
end

function build_many_agents_for_comparison(n_actions::Int)
    return Dict(
        "EXP3" => () -> begin
            l = GameLab.EXP3Learners.EXP3(0.07, 0.07, n_actions)
            TrainableAgent("EXP3", l, GameLab.EXP3Learners.EXP3State(l), :bandit, n_actions)
        end,

        "UCB1" => () -> begin
            l = GameLab.UCBLearners.UCB1(2.0, n_actions)
            TrainableAgent("UCB1", l, GameLab.UCBLearners.UCB1State(l), :bandit, n_actions)
        end,

        "Thompson" => () -> begin
            l = GameLab.ThompsonLearners.GaussianThompson(0.0, 1.0, 1.0, n_actions)
            TrainableAgent("Thompson", l, GameLab.ThompsonLearners.GaussianThompsonState(l), :bandit, n_actions)
        end,

        "Hedge" => () -> begin
            l = GameLab.HedgeLearners.Hedge(0.08, n_actions)
            TrainableAgent("Hedge", l, GameLab.HedgeLearners.HedgeState(l), :full_information, n_actions)
        end,

        "FTPL" => () -> begin
            l = GameLab.FTPLLearners.FTPL(0.2, n_actions)
            TrainableAgent("FTPL", l, GameLab.FTPLLearners.FTPLState(l), :full_information, n_actions)
        end,

        "FTRL" => () -> begin
            l = GameLab.FTRLLearners.EntropicFTRL(0.08, n_actions)
            TrainableAgent("FTRL", l, GameLab.FTRLLearners.EntropicFTRLState(l), :full_information, n_actions)
        end,
    )
end

# ------------------------------------------------------------------------------
# Tournament runner
# ------------------------------------------------------------------------------

function round_robin_bertrand(game; rounds::Int = 10_000, seed::Int = 42)
    n_actions = length(game.price_grid)
    factories = build_many_agents_for_comparison(n_actions)
    names = collect(keys(factories))

    results = Dict{Tuple{String,String},Tuple{Float64,Float64}}()

    for a1 in names, a2 in names
        rng = MersenneTwister(seed)
        agent1 = factories[a1]()
        agent2 = factories[a2]()
        agents = [agent1, agent2]

        run_training!(game, agents; rounds = rounds, rng = rng, log_every = rounds)

        avg1 = mean(agent1.reward_trace)
        avg2 = mean(agent2.reward_trace)
        results[(a1, a2)] = (avg1, avg2)

        @printf("%-10s vs %-10s -> (%8.4f, %8.4f)\n", a1, a2, avg1, avg2)
    end

    return results
end

# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------

function main()
    rng = MersenneTwister(7)

    game = build_homogeneous_bertrand_game()
    n_actions = length(game.price_grid)

    println("price grid = ", game.price_grid)
    println("n_actions  = ", n_actions)
    println()

    agents = build_agents(n_actions)

    run_training!(game, agents; rounds = 20_000, rng = rng, log_every = 2_000)
    summarize_agents(game, agents)

    println("=== round-robin comparison ===")
    round_robin_bertrand(game; rounds = 5_000, seed = 11)

    return nothing
end

main()