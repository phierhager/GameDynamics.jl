module FirstPriceAuctions

using Random

using ..Kernel
using ..RuntimeRecords
using ..Enumerative
using ..Families

export FirstPriceAuctionGame
export FirstPriceAuctionState

export reserve_price
export valuations
export winning_bid
export winner
export realized_payoffs

"""
One-shot simultaneous first-price auction.

Actions are indexed bids:
- action `1` means bid `0`
- action `2` means bid `1`
- ...
- action `k` means bid `k-1`
"""
struct FirstPriceAuctionGame{N} <: Kernel.AbstractGame{N,NTuple{N,Float64}}
    action_sizes::NTuple{N,Int}
    valuations::NTuple{N,Float64}
    reserve_price::Int

    function FirstPriceAuctionGame(action_sizes::NTuple{N,Int},
                                   valuations::NTuple{N,<:Real};
                                   reserve_price::Integer = 0) where {N}
        @inbounds for i in 1:N
            action_sizes[i] > 0 || throw(ArgumentError("Each player must have at least one action."))
        end
        reserve_price >= 0 || throw(ArgumentError("reserve_price must be nonnegative."))

        vals = ntuple(i -> Float64(valuations[i]), N)
        return new{N}(action_sizes, vals, Int(reserve_price))
    end
end

"""
Terminal state stores realized outcome.

- `played == false`: simultaneous move pending
- `played == true`: terminal
- `winner_id == 0`: no sale
"""
struct FirstPriceAuctionState{N,R} <: Kernel.AbstractState
    played::Bool
    winner_id::Int
    winning_bid::Int
    reward::Union{Nothing,R}
end

FirstPriceAuctionState{N}() where {N} =
    FirstPriceAuctionState{N,NTuple{N,Float64}}(false, 0, 0, nothing)

reserve_price(g::FirstPriceAuctionGame) = g.reserve_price
valuations(g::FirstPriceAuctionGame) = g.valuations
winning_bid(s::FirstPriceAuctionState) = s.winning_bid
winner(s::FirstPriceAuctionState) = s.winner_id
realized_payoffs(s::FirstPriceAuctionState) = s.reward

Families.game_family(::Type{<:FirstPriceAuctionGame}) = Families.NormalFormFamily()

Kernel.record_type(::Type{<:FirstPriceAuctionGame{N}}) where {N} =
    RuntimeRecords.JointBanditRecord{NTuple{N,Int},NTuple{N,Float64}}

Kernel.action_mode(::Type{<:FirstPriceAuctionGame}) = Kernel.IndexedActions()
Kernel.has_action_mask(::Type{<:FirstPriceAuctionGame}) = true

Kernel.init_state(::FirstPriceAuctionGame{N},
                  rng::AbstractRNG = Random.default_rng()) where {N} =
    FirstPriceAuctionState{N}()

Kernel.node_kind(::FirstPriceAuctionGame, s::FirstPriceAuctionState) =
    s.played ? Kernel.TERMINAL : Kernel.SIMULTANEOUS

Kernel.active_players(::FirstPriceAuctionGame{N}, s::FirstPriceAuctionState) where {N} =
    Base.OneTo(N)

Kernel.legal_actions(g::FirstPriceAuctionGame,
                     s::FirstPriceAuctionState,
                     player::Int) =
    Base.OneTo(g.action_sizes[player])

Kernel.indexed_action_count(g::FirstPriceAuctionGame, player::Int) =
    g.action_sizes[player]

Kernel.legal_action_mask(g::FirstPriceAuctionGame,
                         s::FirstPriceAuctionState,
                         player::Int) =
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

function _first_price_reward(g::FirstPriceAuctionGame{N},
                             profile::NTuple{N,Int},
                             winner_id::Int) where {N}
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

    next_state = FirstPriceAuctionState{N,typeof(reward)}(
        true,
        winner_id,
        winner_id == 0 ? 0 : bids[winner_id],
        reward,
    )
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

function Enumerative.terminal_payoffs(g::FirstPriceAuctionGame{N},
                                      s::FirstPriceAuctionState{N}) where {N}
    s.played || throw(ArgumentError("terminal_payoffs requires a terminal state."))
    isnothing(s.reward) && throw(ArgumentError("Terminal state does not store reward."))
    return s.reward
end

end