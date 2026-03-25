module Graphical

using ..Classification

export LocalPayoffGame
export local_payoff
export total_payoff
export neighbors

struct LocalPayoffGame{N,Adj,F}
    adjacency::Adj
    local_payoffs::F
end

neighbors(g::LocalPayoffGame, player::Int) = g.adjacency[player]

function _local_profile(profile::NTuple{N,Int}, idxs) where {N}
    return ntuple(k -> profile[idxs[k]], length(idxs))
end

function local_payoff(g::LocalPayoffGame{N}, player::Int, profile::NTuple{N,Int}) where {N}
    idxs = neighbors(g, player)
    lp = _local_profile(profile, idxs)
    return Float64(g.local_payoffs(player, lp))
end

function total_payoff(g::LocalPayoffGame{N}, profile::NTuple{N,Int}) where {N}
    return ntuple(i -> local_payoff(g, i, profile), N)
end

Classification.is_graphical_game(::LocalPayoffGame) = true
Classification.is_network_game(::LocalPayoffGame) = true

end