module ApproxSolverCommon

using ..TabularTraits
using ..TabularMatrixGames
using ..TabularExtensiveTrees

export require_supported_2p_tree_model
export require_tabular_2p_matrix_game
export require_solver_grade_model
export reset_solver!
export run_solver!
export average_policy!
export current_policy!

@inline function require_solver_grade_model(model)
    TabularTraits.is_solver_grade(model) ||
        throw(ArgumentError("This solver requires a solver-grade tabular model, but received a $(TabularTraits.model_role(model))."))
    return nothing
end

@inline function require_supported_2p_tree_model(model::TabularExtensiveTrees.TabularExtensiveTree)
    require_solver_grade_model(model)
    model.n_players == 2 ||
        throw(ArgumentError("This approximate solver currently supports 2-player tabular extensive trees only."))
    TabularTraits.supports_approx_solvers(model) ||
        throw(ArgumentError("This tabular extensive tree is not on an approximate-solver path."))
    return nothing
end

@inline function require_tabular_2p_matrix_game(game::TabularMatrixGames.TabularMatrixGame)
    require_solver_grade_model(game)
    game.n_actions_p1 > 0 || throw(ArgumentError("Tabular matrix game must have at least one action for player 1."))
    game.n_actions_p2 > 0 || throw(ArgumentError("Tabular matrix game must have at least one action for player 2."))
    return nothing
end

function reset_solver!(ws)
    throw(MethodError(reset_solver!, (ws,)))
end

function run_solver!(game, ws; n_iter::Int = 1_000)
    throw(MethodError(run_solver!, (game, ws)))
end

function average_policy!(dest, ws, player::Int)
    throw(MethodError(average_policy!, (dest, ws, player)))
end

function current_policy!(dest, ws, player::Int)
    throw(MethodError(current_policy!, (dest, ws, player)))
end

end