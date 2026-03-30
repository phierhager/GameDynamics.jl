module HomogeneousBertrand

using Random

using ..Kernel
using ..RuntimeRecords
using ..Enumerative
using ..Families

export HomogeneousBertrandGame
export HomogeneousBertrandState

export realized_prices
export realized_quantity
export realized_profits
export demand_at_price

"""
Single-stage homogeneous Bertrand pricing game.

- players simultaneously choose indexed prices from `price_grid`
- lowest price captures the market
- ties split demand equally
- reward is profit `(p - c) * q`
"""
struct HomogeneousBertrandGame{N,F} <: Kernel.AbstractGame{N,NTuple{N,Float64}}
    price_grid::Vector{Float64}
    marginal_costs::NTuple{N,Float64}
    demand_curve::F

    function HomogeneousBertrandGame(price_grid::AbstractVector{<:Real},
                                     marginal_costs::NTuple{N,<:Real},
                                     demand_curve::F) where {N,F}
        isempty(price_grid) && throw(ArgumentError("price_grid must be nonempty."))

        pg = Float64.(collect(price_grid))
        @inbounds for i in eachindex(pg)
            isfinite(pg[i]) || throw(ArgumentError("price_grid must contain only finite values."))
        end

        mc = ntuple(i -> Float64(marginal_costs[i]), N)

        return new{N,F}(pg, mc, demand_curve)
    end
end

"""
Single-stage state.

When `played == false`, the simultaneous move has not occurred yet.
When `played == true`, the state is terminal and stores the realized outcome.
"""
struct HomogeneousBertrandState{N,R} <: Kernel.AbstractState
    played::Bool
    market_price::Union{Nothing,Float64}
    prices::Union{Nothing,NTuple{N,Float64}}
    quantities::Union{Nothing,NTuple{N,Float64}}
    reward::Union{Nothing,R}
end

HomogeneousBertrandState{N}() where {N} =
    HomogeneousBertrandState{N,NTuple{N,Float64}}(false, nothing, nothing, nothing, nothing)

Families.game_family(::Type{<:HomogeneousBertrandGame}) = Families.SimultaneousOneShotFamily()

Kernel.record_type(::Type{<:HomogeneousBertrandGame{N}}) where {N} =
    RuntimeRecords.JointBanditRecord{NTuple{N,Int},NTuple{N,Float64}}

Kernel.action_mode(::Type{<:HomogeneousBertrandGame}) = Kernel.IndexedActions()
Kernel.has_action_mask(::Type{<:HomogeneousBertrandGame}) = true

Kernel.init_state(::HomogeneousBertrandGame{N},
                  rng::AbstractRNG = Random.default_rng()) where {N} =
    HomogeneousBertrandState{N}()

Kernel.node_kind(::HomogeneousBertrandGame, s::HomogeneousBertrandState) =
    s.played ? Kernel.TERMINAL : Kernel.SIMULTANEOUS

Kernel.active_players(::HomogeneousBertrandGame{N}, s::HomogeneousBertrandState) where {N} =
    Base.OneTo(N)

Kernel.legal_actions(g::HomogeneousBertrandGame,
                     s::HomogeneousBertrandState,
                     player::Int) =
    Base.OneTo(length(g.price_grid))

Kernel.indexed_action_count(g::HomogeneousBertrandGame, player::Int) =
    length(g.price_grid)

Kernel.legal_action_mask(g::HomogeneousBertrandGame,
                         s::HomogeneousBertrandState,
                         player::Int) =
    ntuple(_ -> true, length(g.price_grid))

Kernel.observe(::HomogeneousBertrandGame, s::HomogeneousBertrandState, player::Int) = nothing

@inline _prices_from_profile(g::HomogeneousBertrandGame{N}, profile::NTuple{N,Int}) where {N} =
    ntuple(i -> g.price_grid[profile[i]], N)

demand_at_price(g::HomogeneousBertrandGame, p::Real) =
    max(0.0, Float64(g.demand_curve(Float64(p))))

function _homogeneous_outcome(g::HomogeneousBertrandGame{N},
                              profile::NTuple{N,Int}) where {N}
    prices = _prices_from_profile(g, profile)
    pstar = minimum(prices)
    qtotal = demand_at_price(g, pstar)

    winners = Int[]
    @inbounds for i in 1:N
        prices[i] == pstar && push!(winners, i)
    end

    share = qtotal / length(winners)
    quantities = ntuple(i -> (i in winners ? share : 0.0), N)
    profits = ntuple(i -> (prices[i] - g.marginal_costs[i]) * quantities[i], N)

    return prices, pstar, quantities, profits
end

realized_prices(s::HomogeneousBertrandState) = s.prices
realized_quantity(s::HomogeneousBertrandState) = s.quantities
realized_profits(s::HomogeneousBertrandState) = s.reward

function Kernel.step(g::HomogeneousBertrandGame{N},
                     s::HomogeneousBertrandState{N},
                     a::Kernel.JointAction,
                     rng::AbstractRNG = Random.default_rng()) where {N}
    s.played && throw(ArgumentError("Cannot step from terminal homogeneous Bertrand state."))
    Kernel.validate_joint_action(g, s, a)

    profile = Tuple(a)
    prices, pstar, quantities, reward = _homogeneous_outcome(g, profile)

    ns = HomogeneousBertrandState{N,typeof(reward)}(
        true,
        pstar,
        prices,
        quantities,
        reward,
    )
    return ns, reward
end

function Kernel.make_record(g::HomogeneousBertrandGame{N},
                            state::HomogeneousBertrandState{N},
                            action::Kernel.JointAction,
                            next_state::HomogeneousBertrandState{N},
                            reward;
                            done::Bool = Kernel.is_terminal(g, next_state)) where {N}
    return RuntimeRecords.JointBanditRecord(Tuple(action), reward, done)
end

function Enumerative.transition_kernel(g::HomogeneousBertrandGame{N},
                                       s::HomogeneousBertrandState{N},
                                       a::Kernel.JointAction) where {N}
    ns, r = Kernel.step(g, s, a)
    return ((ns, 1.0, r),)
end

function Enumerative.terminal_payoffs(g::HomogeneousBertrandGame{N},
                                      s::HomogeneousBertrandState{N}) where {N}
    s.played || throw(ArgumentError("terminal_payoffs requires a terminal state."))
    isnothing(s.reward) && throw(ArgumentError("Terminal state does not store reward."))
    return s.reward
end

end