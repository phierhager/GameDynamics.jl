module HomogeneousBertrand

Kernel.legal_actions(g::HomogeneousBertrandGame, s::HomogeneousBertrandState, player::Int) =
    Base.OneTo(length(g.price_grid))

Kernel.indexed_action_count(g::HomogeneousBertrandGame, player::Int) = length(g.price_grid)

Kernel.legal_action_mask(g::HomogeneousBertrandGame, s::HomogeneousBertrandState, player::Int) =
    ntuple(_ -> true, length(g.price_grid))

Kernel.observe(::HomogeneousBertrandGame, s::HomogeneousBertrandState, player::Int) = nothing

@inline _prices_from_profile(g::HomogeneousBertrandGame{N}, profile::NTuple{N,Int}) where {N} =
    ntuple(i -> g.price_grid[profile[i]], N)

demand_at_price(g::HomogeneousBertrandGame, p::Real) = max(0.0, Float64(g.demand_curve(Float64(p))))

function _homogeneous_outcome(g::HomogeneousBertrandGame{N}, profile::NTuple{N,Int}) where {N}
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
    return pstar, quantities, profits
end

realized_quantity(s::HomogeneousBertrandState) = s.quantities
realized_profits(s::HomogeneousBertrandState) = s.reward

function Kernel.step(g::HomogeneousBertrandGame{N},
                     s::HomogeneousBertrandState{N},
                     a::Kernel.JointAction,
                     rng::AbstractRNG = Random.default_rng()) where {N}
    s.played && throw(ArgumentError("Cannot step from terminal Bertrand state."))
    Kernel.validate_joint_action(g, s, a)

    profile = Tuple(a)
    pstar, quantities, reward = _homogeneous_outcome(g, profile)
    ns = HomogeneousBertrandState{N,typeof(reward)}(true, pstar, quantities, reward)
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

function Enumerative.terminal_payoffs(g::HomogeneousBertrandGame{N}, s::HomogeneousBertrandState{N}) where {N}
    s.played || throw(ArgumentError("terminal_payoffs requires a terminal state."))
    isnothing(s.reward) && throw(ArgumentError("Terminal state does not store reward."))
    return s.reward
end

end