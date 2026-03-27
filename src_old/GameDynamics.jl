# src/GameDynamics.jl
module GameDynamics

# =========================================================
# 1. The Core API and Engine
# =========================================================
module Core
    include("core/api.jl")
    include("core/engine.jl")
end

# Export the core module so users can access the interface
export Core 

# =========================================================
# 2. Environments Submodule
# =========================================================
module Envs
    using ..Core          # Brings in AbstractGame, etc.
    using StaticArrays

    # Import the API functions so we can extend them
    import ..Core: legal_actions, transition, reward, observe, is_terminal, player_to_move, chance_outcomes, counterfactual_rewards

    include("envs/matrix_games.jl")
    include("envs/gridworlds.jl")
    include("envs/extensive_games.jl")

    # Export specific environments
    export NormalFormGame, MeanFieldGame, PolymatrixGame, GridWorld, Nim, Chicken, PrisonersDilemma
end

# =========================================================
# 3. Agents Submodule
# =========================================================
module Agents
    using ..Core
    using StaticArrays

    import ..Core: act, learn, ContinuousSpace, clip

    include("agents/tabular.jl")
    include("agents/continuous.jl")
    include("agents/population.jl")
    include("agents/regret.jl")

    export QLearningAgent, ContinuousRandomAgent, LinearGaussianAgent, ReplicatorAgent
    export RegretMatchingAgent, FictitiousPlayAgent, InternalRegretAgent
end

module Solvers
    using ..Core
    using ..Envs

    include("solvers/linear_programming.jl")

    export compute_zero_sum_nash, compute_max_welfare_ce
end

end 