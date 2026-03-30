module FirstPriceAuctions
Kernel.legal_actions(g::FirstPriceAuctionGame, s::FirstPriceAuctionState, player::Int) =
    Base.OneTo(g.action_sizes[player])

Kernel.indexed_action_count(g::FirstPriceAuctionGame, player::Int) = g.action_sizes[player]

Kernel.legal_action_mask(g::FirstPriceAuctionGame, s::FirstPriceAuctionState, player::Int) =
    ntuple(_ -> true, g.action_sizes[player])

Kernel.observe(::FirstPriceAuctionGame, s::FirstPriceAuctionState, player::Int) = nothing

@inline _bid_from_action(a::Int) = a - 1

function _top_bidders(profile::NTuple{N,Int}) where {N}
    bids = ntuple(i -> _bid_from_action(profile[i]), N)
    maxbid = maximum(bids)
    tops = Int[]
    @inbounds for i in 1:N
        bids[i] == maxbid && push!(tops, i)
    end
    return bids, maxbid, Tuple(tops)
end

function _first_price_reward(g::FirstPriceAuctionGame{N}, profile::NTuple{N,Int}, winner_id::Int) where {N}
    bids = ntuple(i -> _bid_from_action(profile[i]), N)
    if winner_id == 0
        return ntuple(_ -> 0.0, N)
    end
    winbid = bids[winner_id]
    return ntuple(i -> i == winner_id ? (g.valuations[i] - Float64(winbid)) : 0.0, N)
end

function Kernel.step(g::FirstPriceAuctionGame{N},
                     s::FirstPriceAuctionState{N},
                     a::Kernel.JointAction,
                     rng::AbstractRNG = Random.default_rng()) where {N}
    s.played && throw(ArgumentError("Cannot step from a terminal first-price auction state."))
    Kernel.validate_joint_action(g, s, a)

    profile = Tuple(a)
    bids, maxbid, tops = _top_bidders(profile)

    winner_id = (maxbid < g.reserve_price) ? 0 : rand(rng, tops)
    reward = _first_price_reward(g, profile, winner_id)

    next_state = FirstPriceAuctionState{N,typeof(reward)}(true, winner_id, winner_id == 0 ? 0 : bids[winner_id], reward)
    return next_state, reward
end

function Kernel.make_record(g::FirstPriceAuctionGame{N},
                            state::FirstPriceAuctionState{N},
                            action::Kernel.JointAction,
                            next_state::FirstPriceAuctionState{N},
                            reward;
                            done::Bool = Kernel.is_terminal(g, next_state)) where {N}
    return RuntimeRecords.JointBanditRecord(Tuple(action), reward, done)
end

function Enumerative.transition_kernel(g::FirstPriceAuctionGame{N},
                                       s::FirstPriceAuctionState{N},
                                       a::Kernel.JointAction) where {N}
    s.played && throw(ArgumentError("Cannot enumerate transitions from terminal state."))
    Kernel.validate_joint_action(g, s, a)

    profile = Tuple(a)
    bids, maxbid, tops = _top_bidders(profile)

    if maxbid < g.reserve_price
        reward = ntuple(_ -> 0.0, N)
        ns = FirstPriceAuctionState{N,typeof(reward)}(true, 0, 0, reward)
        return ((ns, 1.0, reward),)
    end

    return ntuple(k -> begin
        w = tops[k]
        reward = _first_price_reward(g, profile, w)
        ns = FirstPriceAuctionState{N,typeof(reward)}(true, w, bids[w], reward)
        (ns, 1.0 / length(tops), reward)
    end, length(tops))
end

function Enumerative.terminal_payoffs(g::FirstPriceAuctionGame{N}, s::FirstPriceAuctionState{N}) where {N}
    s.played || throw(ArgumentError("terminal_payoffs requires a terminal state."))
    isnothing(s.reward) && throw(ArgumentError("Terminal state does not store reward."))
    return s.reward
end

end