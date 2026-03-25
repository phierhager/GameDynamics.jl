module RepeatedAuctions

using ..Domains
using ..LearningInterfaces
using ..LearningFeedback

export RepeatedFirstPriceAuction
export AuctionRoundContext

export AuctionBanditFeedback
export AuctionFullInformationFeedback

export opponent_threshold
export chosen_action
export realized_utility
export observation
export utility_vector

export realized_auction_feedback
export full_information_auction_feedback
export run_auction_round!

struct RepeatedFirstPriceAuction{T,V<:AbstractVector{T}} <: LearningInterfaces.AbstractLearningContext
    bid_grid::V
    n_players::Int
end

function RepeatedFirstPriceAuction(bid_grid::AbstractVector{T}, n_players::Int) where {T}
    n_players > 0 || throw(ArgumentError("n_players must be positive."))
    isempty(bid_grid) && throw(ArgumentError("bid_grid must be nonempty."))
    return RepeatedFirstPriceAuction{T,typeof(bid_grid)}(bid_grid, n_players)
end

struct AuctionRoundContext{T,V<:AbstractVector{T}} <: LearningInterfaces.AbstractLearningContext
    values::V
end

function AuctionRoundContext(values::AbstractVector{T}) where {T}
    isempty(values) && throw(ArgumentError("values must be nonempty."))
    return AuctionRoundContext{T,typeof(values)}(values)
end

struct AuctionBanditFeedback{A,T,O<:LearningFeedback.AbstractObservation} <: LearningFeedback.AbstractFeedback
    chosen_action::A
    value::T
    won::Bool
    payment::T
    realized_utility::T
    observation::O
end

struct AuctionFullInformationFeedback{A,T,V,O<:LearningFeedback.AbstractObservation} <: LearningFeedback.AbstractFeedback
    chosen_action::A
    value::T
    won::Bool
    payment::T
    realized_utility::T
    utility_vector::V
    observation::O
end

Domains.domain_family(::RepeatedFirstPriceAuction) = :repeated_auction

chosen_action(f::AuctionBanditFeedback) = f.chosen_action
chosen_action(f::AuctionFullInformationFeedback) = f.chosen_action

realized_utility(f::AuctionBanditFeedback) = f.realized_utility
realized_utility(f::AuctionFullInformationFeedback) = f.realized_utility

observation(f::AuctionBanditFeedback) = f.observation
observation(f::AuctionFullInformationFeedback) = f.observation

utility_vector(f::AuctionFullInformationFeedback) = f.utility_vector

LearningFeedback.is_bandit_feedback(::AuctionBanditFeedback) = true
LearningFeedback.is_full_information(::AuctionFullInformationFeedback) = true

AuctionBanditFeedback(chosen_action, value, won, payment, realized_utility) =
    AuctionBanditFeedback(chosen_action, value, won, payment, realized_utility, LearningFeedback.NoObservation())

AuctionFullInformationFeedback(chosen_action, value, won, payment, realized_utility, utility_vector) =
    AuctionFullInformationFeedback(chosen_action, value, won, payment, realized_utility, utility_vector, LearningFeedback.NoObservation())

function opponent_threshold(bids::AbstractVector, player::Int)
    1 <= player <= length(bids) || throw(BoundsError(bids, player))
    best = typemin(eltype(bids))
    @inbounds for i in eachindex(bids)
        i == player && continue
        if bids[i] > best
            best = bids[i]
        end
    end
    return best
end

@inline function _winner_index_tie_first(bids::AbstractVector)
    best_i = 1
    best_v = bids[1]
    @inbounds for i in 2:length(bids)
        if bids[i] > best_v
            best_v = bids[i]
            best_i = i
        end
    end
    return best_i
end

@inline _utility(value, won::Bool, payment) = won ? (value - payment) : zero(payment)

function realized_auction_feedback(env::RepeatedFirstPriceAuction,
                                   ctx::AuctionRoundContext,
                                   bids::AbstractVector,
                                   player::Int,
                                   action_idx::Int)
    1 <= action_idx <= length(env.bid_grid) || throw(BoundsError(env.bid_grid, action_idx))
    winner = _winner_index_tie_first(bids)
    won = winner == player
    payment = won ? bids[player] : zero(eltype(env.bid_grid))
    value = ctx.values[player]
    u = _utility(value, won, payment)
    return AuctionBanditFeedback(action_idx, value, won, payment, u)
end

function full_information_auction_feedback(env::RepeatedFirstPriceAuction,
                                           ctx::AuctionRoundContext,
                                           bids::AbstractVector,
                                           player::Int,
                                           action_idx::Int)
    1 <= action_idx <= length(env.bid_grid) || throw(BoundsError(env.bid_grid, action_idx))

    winner = _winner_index_tie_first(bids)
    won = winner == player
    payment = won ? bids[player] : zero(eltype(env.bid_grid))
    value = ctx.values[player]
    realized_u = _utility(value, won, payment)

    thr = opponent_threshold(bids, player)
    uv = similar(env.bid_grid, promote_type(typeof(value), eltype(env.bid_grid)))

    @inbounds for a in eachindex(env.bid_grid)
        bid = env.bid_grid[a]
        won_cf = bid > thr
        uv[a] = _utility(value, won_cf, bid)
    end

    return AuctionFullInformationFeedback(
        action_idx, value, won, payment, realized_u, uv
    )
end

function run_auction_round!(env::RepeatedFirstPriceAuction,
                            ctx::AuctionRoundContext,
                            action_indices::AbstractVector{<:Integer})
    length(action_indices) == env.n_players ||
        throw(ArgumentError("Action vector length does not match player count."))

    bids = similar(env.bid_grid, env.n_players)
    @inbounds for i in 1:env.n_players
        ai = action_indices[i]
        1 <= ai <= length(env.bid_grid) || throw(BoundsError(env.bid_grid, ai))
        bids[i] = env.bid_grid[ai]
    end

    winner = _winner_index_tie_first(bids)
    payments = similar(bids)
    utilities = similar(bids)

    @inbounds for i in 1:env.n_players
        won = (i == winner)
        payments[i] = won ? bids[i] : zero(eltype(bids))
        utilities[i] = _utility(ctx.values[i], won, payments[i])
    end

    return bids, winner, payments, utilities
end

end