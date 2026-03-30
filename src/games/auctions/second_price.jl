module SecondPriceAuctions

using Random

using ..Kernel
using ..RuntimeRecords
using ..Enumerative
using ..Families

export SecondPriceAuctionGame
export SecondPriceAuctionState

export reserve_price
export valuations
export clearing_price
export winner
export realized_payoffs

"""
One-shot simultaneous second-price auction.

Actions are indexed bids:
- action `1` means bid `0`
- action `2` means bid `1`
- ...
- action `k` means bid `k-1`
"""
struct SecondPriceAuctionGame{N} <: Kernel.AbstractGame{N,NTuple{N,Float64}}
    action_sizes::NTuple{N,Int}
    valuations::NTuple{N,Float64}
    reserve_price::Int

    function SecondPriceAuctionGame(action_sizes::NTuple{N,Int},
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
struct SecondPriceAuctionState{N,R} <: Kernel.AbstractState
    played::Bool
    winner_id::Int
    price::Float64
    reward::Union{Nothing,R}
end

SecondPriceAuctionState{N}() where {N} =
    SecondPriceAuctionState{N,NTuple{N,Float64}}(false, 0, 0.0, nothing)

reserve_price(g::SecondPriceAuctionGame) = g.reserve_price
valuations(g::SecondPriceAuctionGame) = g.valuations
clearing_price(s::SecondPriceAuctionState) = s.price
winner(s::SecondPriceAuctionState) = s.winner_id
realized_payoffs(s::SecondPriceAuctionState) = s.reward

Families.game_family(::Type{<:SecondPriceAuctionGame}) = Families.NormalFormFamily()

Kernel.record_type(::Type{<:SecondPriceAuctionGame{N}}) where {N} =
    RuntimeRecords.JointBanditRecord{NTuple{N,Int},NTuple{N,Float64}}

Kernel.action_mode(::Type{<:SecondPriceAuctionGame}) = Kernel.IndexedActions()
Kernel.has_action_mask(::Type{<:SecondPriceAuctionGame}) = true

Kernel.init_state(::SecondPriceAuctionGame{N},
                  rng::AbstractRNG = Random.default_rng()) where {N} =
    SecondPriceAuctionState{N}()

Kernel.node_kind(::SecondPriceAuctionGame, s::SecondPriceAuctionState) =
    s.played ? Kernel.TERMINAL : Kernel.SIMULTANEOUS

Kernel.active_players(::SecondPriceAuctionGame{N}, s::SecondPriceAuctionState) where {N} =
    Base.OneTo(N)

Kernel.legal_actions(g::SecondPriceAuctionGame,
                     s::SecondPriceAuctionState,
                     player::Int) =
    Base.OneTo(g.action_sizes[player])

Kernel.indexed_action_count(g::SecondPriceAuctionGame, player::Int) =
    g.action_sizes[player]

Kernel.legal_action_mask(g::SecondPriceAuctionGame,
                         s::SecondPriceAuctionState,
                         player::Int) =
    ntuple(_ -> true, g.action_sizes[player])

Kernel.observe(::SecondPriceAuctionGame, s::SecondPriceAuctionState, player::Int) = nothing

@inline _bid_from_action(a::Int) = a - 1

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

function _second_price_reward(g::SecondPriceAuctionGame{N},
                              profile::NTuple{N,Int},
                              winner_id::Int,
                              price::Float64) where {N}
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

function Enumerative.terminal_payoffs(g::SecondPriceAuctionGame{N},
                                      s::SecondPriceAuctionState{N}) where {N}
    s.played || throw(ArgumentError("terminal_payoffs requires a terminal state."))
    isnothing(s.reward) && throw(ArgumentError("Terminal state does not store reward."))
    return s.reward
end

end