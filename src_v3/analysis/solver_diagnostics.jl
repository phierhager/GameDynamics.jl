module SolverDiagnostics

using ..TabularMatrixGames
using ..ApproxSolverCommon
using ..MatrixGameAnalysis

export current_policy_value
export average_policy_value

function current_policy_value(game::TabularMatrixGames.TabularMatrixGame, ws)
    x = Vector{Float64}(undef, game.n_actions_p1)
    y = Vector{Float64}(undef, game.n_actions_p2)

    ApproxSolverCommon.current_policy!(x, ws, 1)
    ApproxSolverCommon.current_policy!(y, ws, 2)

    return MatrixGameAnalysis.expected_payoff(game, x, y)
end

function average_policy_value(game::TabularMatrixGames.TabularMatrixGame, ws)
    x = Vector{Float64}(undef, game.n_actions_p1)
    y = Vector{Float64}(undef, game.n_actions_p2)

    ApproxSolverCommon.average_policy!(x, ws, 1)
    ApproxSolverCommon.average_policy!(y, ws, 2)

    return MatrixGameAnalysis.expected_payoff(game, x, y)
end

end