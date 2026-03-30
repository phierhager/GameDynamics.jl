module SolverPrepare

using ..Kernel
using ..SolverAPI
using ..NormalForm
using ..TabularCompile
using ..TabularValidation
using ..TabularMatrixGames
using ..TabularMDPs
using ..TabularMarkovGames
using ..TabularExtensiveTrees
using ..TabularExtensiveGraphs

export prepare_for_solver

function prepare_for_solver end

# ----------------------------------------------------------------------
# Normal-form equilibrium solvers
# ----------------------------------------------------------------------

function prepare_for_solver(::SolverAPI.ZeroSumNash,
                            game::NormalForm.NormalFormGame{2})
    model = TabularCompile.compile_matrix_game(game)
    TabularValidation.require_valid_tabular_model(model)
    return model
end

function prepare_for_solver(::SolverAPI.CorrelatedEquilibrium,
                            game::NormalForm.NormalFormGame)
    return game
end

function prepare_for_solver(::SolverAPI.CoarseCorrelatedEquilibrium,
                            game::NormalForm.NormalFormGame)
    return game
end

# ----------------------------------------------------------------------
# MDP dynamic programming
# ----------------------------------------------------------------------

function prepare_for_solver(::SolverAPI.ValueIteration,
                            game::Kernel.AbstractGame;
                            states)
    model = TabularCompile.compile_mdp(game, states)
    TabularValidation.require_valid_tabular_model(model)
    return model
end

function prepare_for_solver(::SolverAPI.ValueIteration,
                            model::TabularMDPs.TabularMDP)
    TabularValidation.require_valid_tabular_model(model)
    return model
end

# ----------------------------------------------------------------------
# Zero-sum Markov game
# ----------------------------------------------------------------------

function prepare_for_solver(::SolverAPI.ShapleyIteration,
                            game::Kernel.AbstractGame;
                            states)
    model = TabularCompile.compile_zero_sum_markov_game(game, states)
    TabularValidation.require_valid_tabular_model(model)
    return model
end

function prepare_for_solver(::SolverAPI.ShapleyIteration,
                            model::TabularMarkovGames.TabularZeroSumMarkovGame)
    TabularValidation.require_valid_tabular_model(model)
    return model
end

# ----------------------------------------------------------------------
# Extensive-form approximate solvers
# ----------------------------------------------------------------------

function prepare_for_solver(::SolverAPI.CFR,
                            game::Kernel.AbstractGame;
                            root_state = Kernel.init_state(game),
                            infoset_type = Any,
                            label_type = Any,
                            max_nodes::Int = 1_000_000,
                            expand_simultaneous::Bool = false)
    model = TabularCompile.compile_extensive_tree(
        game,
        root_state;
        infoset_type = infoset_type,
        label_type = label_type,
        max_nodes = max_nodes,
        expand_simultaneous = expand_simultaneous,
    )
    TabularValidation.require_valid_tabular_model(model)
    return model
end

function prepare_for_solver(::SolverAPI.CFR,
                            model::TabularExtensiveTrees.TabularExtensiveTree)
    TabularValidation.require_valid_tabular_model(model)
    return model
end

function prepare_for_solver(::SolverAPI.CFRPlus,
                            game::Kernel.AbstractGame;
                            root_state = Kernel.init_state(game),
                            infoset_type = Any,
                            label_type = Any,
                            max_nodes::Int = 1_000_000,
                            expand_simultaneous::Bool = false)
    model = TabularCompile.compile_extensive_tree(
        game,
        root_state;
        infoset_type = infoset_type,
        label_type = label_type,
        max_nodes = max_nodes,
        expand_simultaneous = expand_simultaneous,
    )
    TabularValidation.require_valid_tabular_model(model)
    return model
end

function prepare_for_solver(::SolverAPI.CFRPlus,
                            model::TabularExtensiveTrees.TabularExtensiveTree)
    TabularValidation.require_valid_tabular_model(model)
    return model
end

function prepare_for_solver(::SolverAPI.MCCFR,
                            game::Kernel.AbstractGame;
                            root_state = Kernel.init_state(game),
                            infoset_type = Any,
                            label_type = Any,
                            max_nodes::Int = 1_000_000,
                            expand_simultaneous::Bool = false)
    model = TabularCompile.compile_extensive_tree(
        game,
        root_state;
        infoset_type = infoset_type,
        label_type = label_type,
        max_nodes = max_nodes,
        expand_simultaneous = expand_simultaneous,
    )
    TabularValidation.require_valid_tabular_model(model)
    return model
end

function prepare_for_solver(::SolverAPI.MCCFR,
                            model::TabularExtensiveTrees.TabularExtensiveTree)
    TabularValidation.require_valid_tabular_model(model)
    return model
end

# ----------------------------------------------------------------------
# Helpful fallbacks
# ----------------------------------------------------------------------

function prepare_for_solver(::SolverAPI.ZeroSumNash,
                            model::TabularMatrixGames.TabularMatrixGame)
    TabularValidation.require_valid_tabular_model(model)
    return model
end

function prepare_for_solver(::SolverAPI.CFR,
                            model::TabularExtensiveGraphs.TabularExtensiveGraph)
    throw(ArgumentError(
        "CFR currently expects a tabular extensive tree, not a graph model."
    ))
end

function prepare_for_solver(::SolverAPI.CFRPlus,
                            model::TabularExtensiveGraphs.TabularExtensiveGraph)
    throw(ArgumentError(
        "CFR+ currently expects a tabular extensive tree, not a graph model."
    ))
end

function prepare_for_solver(::SolverAPI.MCCFR,
                            model::TabularExtensiveGraphs.TabularExtensiveGraph)
    throw(ArgumentError(
        "MCCFR currently expects a tabular extensive tree, not a graph model."
    ))
end

end