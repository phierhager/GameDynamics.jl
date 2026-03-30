module MatrixGames

using ..NormalForm

export MatrixGame
export ZeroSumMatrixGame

"""
Convenience constructor for a 2-player bimatrix normal-form game.
"""
function MatrixGame(A, B)
    size(A) == size(B) || throw(ArgumentError(
        "MatrixGame requires payoff matrices of identical shape."
    ))
    return NormalForm.NormalFormGame((A, B))
end

"""
Convenience constructor for a 2-player zero-sum matrix game.

Player 2 automatically gets payoff `-A`.
"""
ZeroSumMatrixGame(A) = NormalForm.NormalFormGame((A, -A))

end