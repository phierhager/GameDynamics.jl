# ===========================================================================
# src/solvers/linear_programming.jl
# ===========================================================================
using JuMP
using HiGHS
using ..Core
using ..Envs: NormalFormGame

"""
Computes the exact Nash Equilibrium strategy for Player 1 in a 2-Player Zero-Sum Game 
using the classic Maximin Linear Programming formulation.
"""
function compute_zero_sum_nash(game::NormalFormGame{2})
    # In a zero-sum game, we only need Player 1's payoff matrix
    M = game.payoff_matrices[1] 
    n_actions_p1, n_actions_p2 = size(M)
    
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    
    # Variables
    @variable(model, v) # The value of the game
    @variable(model, p[1:n_actions_p1] >= 0) # Player 1's probability distribution
    
    # Constraints
    @constraint(model, sum(p) == 1.0) # Probabilities must sum to 1
    
    # For every pure strategy Player 2 could play (the columns), 
    # Player 1's expected payoff must be AT LEAST `v`.
    @constraint(model, [j=1:n_actions_p2], sum(p[i] * M[i, j] for i in 1:n_actions_p1) >= v)
    
    # Objective: Maximize the guaranteed minimum payoff
    @objective(model, Max, v)
    
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        return value.(p), objective_value(model)
    else
        error("Solver failed to find an optimal equilibrium.")
    end
end


"""
Computes the Correlated Equilibrium (CE) that maximizes Social Welfare.
"""
function compute_max_welfare_ce(game::NormalFormGame{2})
    M1, M2 = game.payoff_matrices
    A1, A2 = size(M1)
    
    model = Model(HiGHS.Optimizer); set_silent(model)
    @variable(model, p[1:A1, 1:A2] >= 0)
    @constraint(model, sum(p) == 1.0)
    
    # Player 1 Rationality: If recommended i, don't prefer j
    for i in 1:A1, j in 1:A1
        if i == j continue end
        @constraint(model, sum(p[i, k] * (M1[i, k] - M1[j, k]) for k in 1:A2) >= 0)
    end
    
    # Player 2 Rationality: If recommended k, don't prefer l
    for k in 1:A2, l in 1:A2
        if k == l continue end
        @constraint(model, sum(p[i, k] * (M2[i, k] - M2[i, l]) for i in 1:A1) >= 0)
    end
    
    @objective(model, Max, sum(p .* (M1 .+ M2)))
    optimize!(model)
    return value.(p)
end