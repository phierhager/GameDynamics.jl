module DifferentiatedBertrand

using Random

using ..Kernel
using ..RuntimeRecords
using ..Enumerative
using ..Families

export DifferentiatedBertrandGame
export DifferentiatedBertrandState

export demand_shares
export realized_prices
export realized_shares
export realized_quantities
export realized_profits

"""
Single-stage differentiated Bertrand pricing game.

Demand shares follow a logit rule:
    attractiveness[i] - price_sensitivity * price[i]

Then total demand is supplied by `demand_curve(prices, shares)`.
"""
struct DifferentiatedBertrandGame{N,F} <: Kernel.AbstractGame{N,NTuple{N,Float64}}
    price_grid::Vector{Float64}
    attractiveness::NTuple{N,Float64}
    price_sensitivity::Float64
    outside_option_utility::Float64
    marginal_costs::NTuple{N,Float64}
    demand_curve::F

    function DifferentiatedBertrandGame(price_grid::AbstractVector{<:Real},
                                        attractiveness::NTuple{N,<:Real},
                                        price_sensitivity::Real,
                                        outside_option_utility::Real,
                                        marginal_costs::NTuple{N,<:Real},
                                        demand_curve::F) where {N,F}
        isempty(price_grid) && throw(ArgumentError("price_grid must be nonempty."))
        price_sensitivity >= 0 || throw(ArgumentError("price_sensitivity must be nonnegative."))

        pg = Float64.(collect(price_grid))
        attr = ntuple(i -> Float64(attractiveness[i]), N)
        mc = ntuple(i -> Float64(marginal_costs[i]), N)

        return new{N,F}(
            pg,
            attr,
            Float64(price_sensitivity),
            Float64(outside_option_utility),
            mc,
            demand_curve,
        )
    end
end

"""
Single-stage state.

When `played == false`, the simultaneous move has not occurred yet.
When `played == true`, the state is terminal and stores the realized outcome.
"""
struct DifferentiatedBertrandState{N,R} <: Kernel.AbstractState
    played::Bool
    prices::Union{Nothing,NTuple{N,Float64}}
    shares::Union{Nothing,NTuple{N,Float64}}
    quantities::Union{Nothing,NTuple{N,Float64}}
    reward::Union{Nothing,R}
end

DifferentiatedBertrandState{N}() where {N} =
    DifferentiatedBertrandState{N,NTuple{N,Float64}}(false, nothing, nothing, nothing, nothing)

Families.game_family(::Type{<:DifferentiatedBertrandGame}) = Families.SimultaneousOneShotFamily()

Kernel.record_type(::Type{<:DifferentiatedBertrandGame{N}}) where {N} =
    RuntimeRecords.JointBanditRecord{NTuple{N,Int},NTuple{N,Float64}}

Kernel.action_mode(::Type{<:DifferentiatedBertrandGame}) = Kernel.IndexedActions()
Kernel.has_action_mask(::Type{<:DifferentiatedBertrandGame}) = true

Kernel.init_state(::DifferentiatedBertrandGame{N},
                  rng::AbstractRNG = Random.default_rng()) where {N} =
    DifferentiatedBertrandState{N}()

Kernel.node_kind(::DifferentiatedBertrandGame, s::DifferentiatedBertrandState) =
    s.played ? Kernel.TERMINAL : Kernel.SIMULTANEOUS

Kernel.active_players(::DifferentiatedBertrandGame{N}, s::DifferentiatedBertrandState) where {N} =
    Base.OneTo(N)

Kernel.legal_actions(g::DifferentiatedBertrandGame,
                     s::DifferentiatedBertrandState,
                     player::Int) =
    Base.OneTo(length(g.price_grid))

Kernel.indexed_action_count(g::DifferentiatedBertrandGame, player::Int) =
    length(g.price_grid)

Kernel.legal_action_mask(g::DifferentiatedBertrandGame,
                         s::DifferentiatedBertrandState,
                         player::Int) =
    ntuple(_ -> true, length(g.price_grid))

Kernel.observe(::DifferentiatedBertrandGame, s::DifferentiatedBertrandState, player::Int) = nothing

@inline _prices_from_profile(g::DifferentiatedBertrandGame{N}, profile::NTuple{N,Int}) where {N} =
    ntuple(i -> g.price_grid[profile[i]], N)

function demand_shares(g::DifferentiatedBertrandGame{N},
                       prices::NTuple{N,Float64}) where {N}
    utils = ntuple(i -> g.attractiveness[i] - g.price_sensitivity * prices[i], N)
    m = maximum(utils)
    expu = ntuple(i -> exp(utils[i] - m), N)
    outside = exp(g.outside_option_utility - m)
    denom = outside + sum(expu)

    return ntuple(i -> expu[i] / denom, N)
end

function _differentiated_outcome(g::DifferentiatedBertrandGame{N},
                                 profile::NTuple{N,Int}) where {N}
    prices = _prices_from_profile(g, profile)
    shares = demand_shares(g, prices)
    total_demand = max(0.0, Float64(g.demand_curve(prices, shares)))
    quantities = ntuple(i -> shares[i] * total_demand, N)
    profits = ntuple(i -> (prices[i] - g.marginal_costs[i]) * quantities[i], N)

    return prices, shares, quantities, profits
end

realized_prices(s::DifferentiatedBertrandState) = s.prices
realized_shares(s::DifferentiatedBertrandState) = s.shares
realized_quantities(s::DifferentiatedBertrandState) = s.quantities
realized_profits(s::DifferentiatedBertrandState) = s.reward

function Kernel.step(g::DifferentiatedBertrandGame{N},
                     s::DifferentiatedBertrandState{N},
                     a::Kernel.JointAction,
                     rng::AbstractRNG = Random.default_rng()) where {N}
    s.played && throw(ArgumentError("Cannot step from terminal differentiated Bertrand state."))
    Kernel.validate_joint_action(g, s, a)

    profile = Tuple(a)
    prices, shares, quantities, reward = _differentiated_outcome(g, profile)

    ns = DifferentiatedBertrandState{N,typeof(reward)}(
        true,
        prices,
        shares,
        quantities,
        reward,
    )
    return ns, reward
end

function Kernel.make_record(g::DifferentiatedBertrandGame{N},
                            state::DifferentiatedBertrandState{N},
                            action::Kernel.JointAction,
                            next_state::DifferentiatedBertrandState{N},
                            reward;
                            done::Bool = Kernel.is_terminal(g, next_state)) where {N}
    return RuntimeRecords.JointBanditRecord(Tuple(action), reward, done)
end

function Enumerative.transition_kernel(g::DifferentiatedBertrandGame{N},
                                       s::DifferentiatedBertrandState{N},
                                       a::Kernel.JointAction) where {N}
    ns, r = Kernel.step(g, s, a)
    return ((ns, 1.0, r),)
end

function Enumerative.terminal_payoffs(g::DifferentiatedBertrandGame{N},
                                      s::DifferentiatedBertrandState{N}) where {N}
    s.played || throw(ArgumentError("terminal_payoffs requires a terminal state."))
    isnothing(s.reward) && throw(ArgumentError("Terminal state does not store reward."))
    return s.reward
end

end