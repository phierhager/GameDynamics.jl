module DifferentiatedBertrand

Kernel.init_state(::DifferentiatedBertrandGame{N}, rng::AbstractRNG = Random.default_rng()) where {N} =
    DifferentiatedBertrandState{N,NTuple{N,Float64}}(false, nothing, nothing, nothing, nothing)

Kernel.node_kind(::DifferentiatedBertrandGame, s::DifferentiatedBertrandState) =
    s.played ? Kernel.TERMINAL : Kernel.SIMULTANEOUS

Kernel.active_players(::DifferentiatedBertrandGame{N}, s::DifferentiatedBertrandState) where {N} = Base.OneTo(N)

Kernel.legal_actions(g::DifferentiatedBertrandGame, s::DifferentiatedBertrandState, player::Int) =
    Base.OneTo(length(g.price_grid))

Kernel.indexed_action_count(g::DifferentiatedBertrandGame, player::Int) = length(g.price_grid)

Kernel.legal_action_mask(g::DifferentiatedBertrandGame, s::DifferentiatedBertrandState, player::Int) =
    ntuple(_ -> true, length(g.price_grid))

Kernel.observe(::DifferentiatedBertrandGame, s::DifferentiatedBertrandState, player::Int) = nothing

@inline _prices_from_profile(g::DifferentiatedBertrandGame{N}, profile::NTuple{N,Int}) where {N} =
    ntuple(i -> g.price_grid[profile[i]], N)

function demand_shares(g::DifferentiatedBertrandGame{N}, prices::NTuple{N,Float64}) where {N}
    utils = ntuple(i -> g.attractiveness[i] - g.price_sensitivity * prices[i], N)
    m = maximum(utils)
    expu = ntuple(i -> exp(utils[i] - m), N)
    outside = exp(g.outside_option_utility - m)
    denom = outside + sum(expu)
    return ntuple(i -> expu[i] / denom, N)
end

function _differentiated_outcome(g::DifferentiatedBertrandGame{N}, profile::NTuple{N,Int}) where {N}
    prices = _prices_from_profile(g, profile)
    shares = demand_shares(g, prices)
    total_demand = max(0.0, Float64(g.demand_curve(prices, shares)))
    quantities = ntuple(i -> shares[i] * total_demand, N)
    profits = ntuple(i -> (prices[i] - g.marginal_costs[i]) * quantities[i], N)
    return prices, shares, quantities, profits
end

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
    ns = DifferentiatedBertrandState{N,typeof(reward)}(true, prices, shares, quantities, reward)
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

function Enumerative.terminal_payoffs(g::DifferentiatedBertrandGame{N}, s::DifferentiatedBertrandState{N}) where {N}
    s.played || throw(ArgumentError("terminal_payoffs requires a terminal state."))
    isnothing(s.reward) && throw(ArgumentError("Terminal state does not store reward."))
    return s.reward
end

end