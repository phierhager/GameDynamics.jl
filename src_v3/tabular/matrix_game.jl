module TabularMatrixGames

using ..TabularTraits

export TabularMatrixGame
export n_actions_p1
export n_actions_p2

struct TabularMatrixGame <: TabularTraits.AbstractTabularNormalFormModel
    payoff_p1::Matrix{Float64}
    payoff_p2::Matrix{Float64}
    n_actions_p1::Int
    n_actions_p2::Int
end

n_actions_p1(g::TabularMatrixGame) = g.n_actions_p1
n_actions_p2(g::TabularMatrixGame) = g.n_actions_p2

TabularTraits.supports_exact_solvers(::TabularMatrixGame) = true
TabularTraits.supports_approx_solvers(::TabularMatrixGame) = true

end