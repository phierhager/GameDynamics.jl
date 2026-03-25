module ApproxSolverCommon

using ..Contracts
using ..CompiledExtensiveModels
using ..CompiledNormalFormModels

export require_supported_2p_tree_model
export require_compiled_2p_matrix_game
export require_solver_grade_model
export reset_solver!
export run_solver!
export average_policy!
export current_policy!

@inline function require_solver_grade_model(model)
    Contracts.is_solver_grade(model) ||
        throw(ArgumentError("This solver requires a solver-grade compiled model, but received a $(Contracts.model_role(model)) model. Use analysis/diagnostic APIs instead."))
    return nothing
end

@inline function require_supported_2p_tree_model(model::CompiledExtensiveModels.CompiledExtensiveGame)
    require_solver_grade_model(model)
    model.n_players == 2 ||
        throw(ArgumentError("This approximate solver currently supports 2-player compiled extensive-form games only."))
    Contracts.supports_approx_solvers(model) ||
        throw(ArgumentError("This compiled extensive-form model is not on an approximate-solver path. Simultaneous-node compiled trees are analysis-only."))
    return nothing
end

@inline function require_compiled_2p_matrix_game(game::CompiledNormalFormModels.CompiledMatrixGame)
    require_solver_grade_model(game)
    game.n_actions_p1 > 0 || throw(ArgumentError("Compiled matrix game must have at least one action for player 1."))
    game.n_actions_p2 > 0 || throw(ArgumentError("Compiled matrix game must have at least one action for player 2."))
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