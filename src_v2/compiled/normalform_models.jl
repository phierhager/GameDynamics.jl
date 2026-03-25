module CompiledNormalFormModels

using ..Compiled
using ..NormalForm

export CompiledMatrixGame
export compile_matrix_game
export n_actions_p1, n_actions_p2

struct CompiledMatrixGame <: Compiled.AbstractCompiledNormalFormModel
    payoff_p1::Matrix{Float64}
    payoff_p2::Matrix{Float64}
    n_actions_p1::Int
    n_actions_p2::Int
end

n_actions_p1(g::CompiledMatrixGame) = g.n_actions_p1
n_actions_p2(g::CompiledMatrixGame) = g.n_actions_p2

function compile_matrix_game(game::NormalForm.NormalFormGame{2})
    U1 = Matrix{Float64}(game.payoffs[1])
    U2 = Matrix{Float64}(game.payoffs[2])
    m, n = size(U1)
    return CompiledMatrixGame(U1, U2, m, n)
end

end