# ===========================================================================
# benchmarks.jl
# Run this from the root directory: `julia benchmarks.jl`
# ===========================================================================

# Include the main package file
include("src/GameDynamics.jl")

using .GameDynamics.Core
using .GameDynamics.Envs
using StaticArrays

import .GameDynamics.Core: act, learn

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
    println("✅ Benchmarks Complete!")
end

# Run the suite
run_benchmarks()