module TabularMarkovGames

using ..TabularTraits

export TabularZeroSumMarkovGame
export n_states
export n_actions

struct TabularZeroSumMarkovGame{SE,S,A1T,A2T} <: TabularTraits.AbstractTabularMarkovModel
    n_states::Int
    n_actions_p1::Vector{Int}
    n_actions_p2::Vector{Int}
    pair_ptr::Vector{Int}          # length n_states + 1
    trans_ptr::Vector{Int}         # length total_joint_actions + 1
    next_state::Vector{Int}
    prob::Vector{Float64}
    reward::Vector{Float64}        # player-1 reward
    state_encoder::SE
    states::Vector{S}
    legal_actions_p1::Vector{A1T}
    legal_actions_p2::Vector{A2T}
end

n_states(m::TabularZeroSumMarkovGame) = m.n_states
n_actions(m::TabularZeroSumMarkovGame, s::Int) = (m.n_actions_p1[s], m.n_actions_p2[s])

TabularTraits.supports_exact_solvers(::TabularZeroSumMarkovGame) = true
TabularTraits.supports_approx_solvers(::TabularZeroSumMarkovGame) = false

end