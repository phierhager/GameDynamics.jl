module Congestion

using ..Classification

export ResourceCongestionGame
export resource_loads
export player_costs
export rosenthal_potential

struct ResourceCongestionGame{F}
    n_players::Int
    n_resources::Int
    cost_fn::F
end

function resource_loads(g::ResourceCongestionGame, profile::NTuple{N,Int}) where {N}
    N == g.n_players || throw(ArgumentError("Profile length does not match player count."))
    loads = zeros(Int, g.n_resources)
    @inbounds for i in 1:N
        loads[profile[i]] += 1
    end
    return loads
end

function player_costs(g::ResourceCongestionGame, profile::NTuple{N,Int}) where {N}
    loads = resource_loads(g, profile)
    return ntuple(i -> Float64(g.cost_fn(profile[i], loads[profile[i]])), N)
end

function rosenthal_potential(g::ResourceCongestionGame, profile::NTuple{N,Int}) where {N}
    loads = resource_loads(g, profile)
    pot = 0.0
    for r in 1:g.n_resources
        for k in 1:loads[r]
            pot += Float64(g.cost_fn(r, k))
        end
    end
    return pot
end

Classification.is_potential_game(::ResourceCongestionGame) = true
Classification.is_anonymous_game(::ResourceCongestionGame) = true

end