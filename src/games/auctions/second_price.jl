module SecondPriceAuctions

function _bid_summary(profile::NTuple{N,Int}) where {N}
    bids = ntuple(i -> _bid_from_action(profile[i]), N)
    sorted = sort(collect(bids); rev = true)
    highest = sorted[1]
    second_highest = N >= 2 ? sorted[2] : 0
    tops = Int[]
    @inbounds for i in 1:N
        bids[i] == highest && push!(tops, i)
    end
    return bids, highest, second_highest, Tuple(tops)
end

function _second_price_reward(g::SecondPriceAuctionGame{N}, profile::NTuple{N,Int}, winner_id::Int, price::Float64) where {N}
    if winner_id == 0
        return ntuple(_ -> 0.0, N)
    end
    return ntuple(i -> i == winner_id ? (g.valuations[i] - price) : 0.0, N)
end

function Kernel.step(g::SecondPriceAuctionGame{N},
                     s::SecondPriceAuctionState{N},
                     a::Kernel.JointAction,
                     rng::AbstractRNG = Random.default_rng()) where {N}
    s.played && throw(ArgumentError("Cannot step from a terminal second-price auction state."))
    Kernel.validate_joint_action(g, s, a)

    profile = Tuple(a)
    bids, highest, second_highest, tops = _bid_summary(profile)

    if highest < g.reserve_price
        reward = ntuple(_ -> 0.0, N)
        ns = SecondPriceAuctionState{N,typeof(reward)}(true, 0, 0.0, reward)
        return ns, reward
    end

    winner_id = rand(rng, tops)
    price = max(g.reserve_price, Float64(second_highest))
    reward = _second_price_reward(g, profile, winner_id, price)
    ns = SecondPriceAuctionState{N,typeof(reward)}(true, winner_id, price, reward)
    return ns, reward
end

function Kernel.make_record(g::SecondPriceAuctionGame{N},
                            state::SecondPriceAuctionState{N},
                            action::Kernel.JointAction,
                            next_state::SecondPriceAuctionState{N},
                            reward;
                            done::Bool = Kernel.is_terminal(g, next_state)) where {N}
    return RuntimeRecords.JointBanditRecord(Tuple(action), reward, done)
end

function Enumerative.transition_kernel(g::SecondPriceAuctionGame{N},
                                       s::SecondPriceAuctionState{N},
                                       a::Kernel.JointAction) where {N}
    s.played && throw(ArgumentError("Cannot enumerate transitions from terminal state."))
    Kernel.validate_joint_action(g, s, a)

    profile = Tuple(a)
    bids, highest, second_highest, tops = _bid_summary(profile)

    if highest < g.reserve_price
        reward = ntuple(_ -> 0.0, N)
        ns = SecondPriceAuctionState{N,typeof(reward)}(true, 0, 0.0, reward)
        return ((ns, 1.0, reward),)
    end

    price = max(g.reserve_price, Float64(second_highest))
    return ntuple(k -> begin
        w = tops[k]
        reward = _second_price_reward(g, profile, w, price)
        ns = SecondPriceAuctionState{N,typeof(reward)}(true, w, price, reward)
        (ns, 1.0 / length(tops), reward)
    end, length(tops))
end

function Enumerative.terminal_payoffs(g::SecondPriceAuctionGame{N}, s::SecondPriceAuctionState{N}) where {N}
    s.played || throw(ArgumentError("terminal_payoffs requires a terminal state."))
    isnothing(s.reward) && throw(ArgumentError("Terminal state does not store reward."))
    return s.reward
end

end