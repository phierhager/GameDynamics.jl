# ===========================================================================
# benchmarks.jl
# Run this from the root directory: `julia benchmarks.jl`
# ===========================================================================

# Include the main package file
include("src/GameDynamics.jl")

using .GameDynamics.Core
using .GameDynamics.Envs
using .GameDynamics.Solvers
using StaticArrays

import .GameDynamics.Core: act, learn, GameRunner, simulate_episode, run_episodes!
using .GameDynamics.Agents: RegretMatchingAgent, FictitiousPlayAgent, InternalRegretAgent
using .GameDynamics.Solvers: compute_zero_sum_nash, compute_max_welfare_ce

# 0. A Dummy Agent for Benchmarking
struct BenchAgent end
act(::BenchAgent, game, obs, valid) = rand(valid)
learn(a::BenchAgent, game, obs, action, reward, next_obs) = a


function run_benchmarks()
    n_episodes = 100_000
    
    println("\n=======================================================")
    println("🚀 RUNNING PERFORMANCE BENCHMARKS (100,000 Episodes Each)")
    println("=======================================================")
    
    # --- 1. GridWorld ---
    grid_game = GridWorld(5)
    agent_tuple_1 = (BenchAgent(),)
    
    println("\n1. GridWorld (MDP, 1-Player, Stateful)")
    simulate_episode(grid_game, 1, agent_tuple_1, max_steps=100) 
    
    @time begin
        for _ in 1:n_episodes
            simulate_episode(grid_game, 1, agent_tuple_1, max_steps=100)
        end
    end
    
    # --- 2. Normal Form RPS ---
    payoff_p1 = @SMatrix [0.0 -1.0 1.0; 1.0 0.0 -1.0; -1.0 1.0 0.0]
    payoff_p2 = @SMatrix [0.0 1.0 -1.0; -1.0 0.0 1.0; 1.0 -1.0 0.0]
    rps_game = NormalFormGame((payoff_p1, payoff_p2))
    
    agent_tuple_2 = (BenchAgent(), BenchAgent())
    
    println("\n2. Rock-Paper-Scissors (Simultaneous, 2-Player, Matrix)")
    simulate_episode(rps_game, 0, agent_tuple_2, max_steps=10)
    
    @time begin
        for _ in 1:n_episodes
            simulate_episode(rps_game, 0, agent_tuple_2, max_steps=10)
        end
    end
    
    # --- 3. Nim ---
    nim_game = Nim()
    agent_tuple_3 = (BenchAgent(), BenchAgent())
    initial_nim_state = (20, 1) 
    
    println("\n3. Nim (Sequential, 2-Player, Turn-Based)")
    simulate_episode(nim_game, initial_nim_state, agent_tuple_3, max_steps=100)
    
    @time begin
        for _ in 1:n_episodes
            simulate_episode(nim_game, initial_nim_state, agent_tuple_3, max_steps=100)
        end
    end


    println("\n=======================================================")
    println("⚖️  EQUILIBRIUM CALCULATION BENCHMARKS")
    println("=======================================================")

    # --- Exact LP Calculation ---
    payoff_p1 = @SMatrix [0.0 -1.0 1.0; 1.0 0.0 -1.0; -1.0 1.0 0.0]
    payoff_p2 = @SMatrix [0.0 1.0 -1.0; -1.0 0.0 1.0; 1.0 -1.0 0.0]
    rps_game = NormalFormGame((payoff_p1, payoff_p2))

    println("\n--- Exact Solver (JuMP + HiGHS) ---")
    # exact_strategy, exact_value = compute_zero_sum_nash(rps_game)
    # println("Exact Nash Equilibrium (Player 1): ", round.(exact_strategy, digits=4))
    # println("Game Value: ", round(exact_value, digits=4))
    println("Exact Nash Equilibrium (Player 1): [0.3333, 0.3333, 0.3333] (Expected)")

    # --- Fictitious Play ---
    println("\n--- Fictitious Play (Converging to NE) ---")
    # Initialize FP agents with zero counts
    fp_p1 = FictitiousPlayAgent{3, 3}(@SVector(zeros(Int, 3)), 1)
    fp_p2 = FictitiousPlayAgent{3, 3}(@SVector(zeros(Int, 3)), 2)

    fp_runner = GameRunner(rps_game, (fp_p1, fp_p2))
    _, final_fp_agents = run_episodes!(fp_runner, 0, n_episodes=50_000, log_every=100_000)

    # Extract the opponent model from Player 2 to see Player 1's historical frequency
    p1_historical_counts = final_fp_agents[2].opp_action_counts
    p1_fp_strategy = p1_historical_counts ./ sum(p1_historical_counts)
    println("FP Empirical Strategy (Player 1): ", round.(p1_fp_strategy, digits=4))


    # --- Regret Matching ---
    println("\n--- Regret Matching (Converging to CCE/NE) ---")
    # Initialize RM agents with zero regrets and zero strategy sum
    rm_p1 = RegretMatchingAgent{3}(@SVector(zeros(3)), @SVector(zeros(3)), 1)
    rm_p2 = RegretMatchingAgent{3}(@SVector(zeros(3)), @SVector(zeros(3)), 2)

    rm_runner = GameRunner(rps_game, (rm_p1, rm_p2))
    _, final_rm_agents = run_episodes!(rm_runner, 0, n_episodes=50_000, log_every=100_000)

    p1_strategy_sum = final_rm_agents[1].strategy_sum
    p1_rm_strategy = p1_strategy_sum ./ sum(p1_strategy_sum)
    println("RM Average Strategy (Player 1):   ", round.(p1_rm_strategy, digits=4))



    println("\n=======================================================")
    println("🚦 SOCIAL DILEMMAS & CORRELATED EQUILIBRIA")
    println("=======================================================")

    # --- Case 1: Chicken ---
    chicken_game = Chicken()
    println("\n--- Environment: CHICKEN ---")

    # 1. Exact CE Solver
    ce_dist = compute_max_welfare_ce(chicken_game)
    println("Exact Max-Welfare CE (Joint Dist):")
    display(round.(ce_dist, digits=3)) 
    # You should see ~0.333 for (1,1), (1,2), and (2,1) and 0.0 for (2,2)

    # 2. Internal Regret Agents
    ir_p1 = InternalRegretAgent{2}(@SMatrix(zeros(2,2)), @SVector(zeros(2)), 1)
    ir_p2 = InternalRegretAgent{2}(@SMatrix(zeros(2,2)), @SVector(zeros(2)), 2)

    chicken_runner = GameRunner(chicken_game, (ir_p1, ir_p2))
    _, final_ir_agents = run_episodes!(chicken_runner, 0, n_episodes=50_000, log_every=100_000)

    p1_ir_strat = final_ir_agents[1].strategy_sum ./ sum(final_ir_agents[1].strategy_sum)
    println("Internal Regret Strategy (P1): ", round.(p1_ir_strat, digits=3))

    # --- Case 2: Prisoner's Dilemma ---
    pd_game = PrisonersDilemma()
    println("\n--- Environment: PRISONER'S DILEMMA ---")

    # 1. Exact CE Solver
    ce_dist_pd = compute_max_welfare_ce(pd_game)
    println("Exact Max-Welfare CE (Joint Dist):")
    display(round.(ce_dist_pd, digits=3))
    # Expected: 1.0 at (2,2) - Defect is the only rational choice!

    # 2. Fictitious Play (just to compare)
    fp_p1_pd = FictitiousPlayAgent{2, 2}(@SVector(zeros(Int, 2)), 1)
    fp_p2_pd = FictitiousPlayAgent{2, 2}(@SVector(zeros(Int, 2)), 2)

    pd_runner = GameRunner(pd_game, (fp_p1_pd, fp_p2_pd))
    _, final_fp_pd = run_episodes!(pd_runner, 0, n_episodes=10_000, log_every=100_000)
    p1_counts = final_fp_pd[2].opp_action_counts
    println("FP Strategy (P1): ", round.(p1_counts ./ sum(p1_counts), digits=3))

    
    println("\n=======================================================")
    println("✅ Benchmarks Complete!")
end

# Run the suite
run_benchmarks()