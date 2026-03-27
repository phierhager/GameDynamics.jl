module TabularMDPs

using ..TabularTraits

export TabularMDP
export n_states
export n_actions
export action_labels

struct TabularMDP{SE,S} <: TabularTraits.AbstractTabularMarkovModel
    n_states::Int
    n_actions_vec::Vector{Int}
    action_ptr::Vector{Int}      # length n_states + 1
    trans_ptr::Vector{Int}       # length total_state_actions + 1
    next_state::Vector{Int}
    prob::Vector{Float64}
    reward::Vector{Float64}
    action_label::Vector{Int}    # actual IndexedActions labels for each state-action slot
    state_encoder::SE
    states::Vector{S}
end

n_states(m::TabularMDP) = m.n_states
n_actions(m::TabularMDP, s::Int) = m.n_actions_vec[s]

function action_labels(m::TabularMDP, s::Int)
    a0 = m.action_ptr[s]
    a1 = m.action_ptr[s + 1] - 1
    a1 < a0 && return Int[]
    return @view m.action_label[a0:a1]
end

TabularTraits.supports_exact_solvers(::TabularMDP) = true
TabularTraits.supports_approx_solvers(::TabularMDP) = false

end