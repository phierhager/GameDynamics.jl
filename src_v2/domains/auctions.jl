module Auctions

using ..Classification

export FirstPriceSealedBid
export SecondPriceSealedBid
export SingleItemAuction
export AuctionOutcome
export auction_outcome
export utility

struct FirstPriceSealedBid end
struct SecondPriceSealedBid end

struct SingleItemAuction{F}
    format::F
    n_bidders::Int
end

struct AuctionOutcome
    winner::Int
    allocation::NTuple{1,Int}
    payments::Vector{Float64}
    winning_bid::Float64
end

function _argmax_tie_first(bids::AbstractVector{<:Real})
    best_i = 1
    best_v = bids[1]
    @inbounds for i in 2:length(bids)
        if bids[i] > best_v
            best_v = bids[i]
            best_i = i
        end
    end
    return best_i, Float64(best_v)
end

function auction_outcome(::FirstPriceSealedBid, bids::AbstractVector{<:Real})
    winner, winbid = _argmax_tie_first(bids)
    payments = zeros(Float64, length(bids))
    payments[winner] = winbid
    return AuctionOutcome(winner, (winner,), payments, winbid)
end

function auction_outcome(::SecondPriceSealedBid, bids::AbstractVector{<:Real})
    winner, winbid = _argmax_tie_first(bids)
    second = 0.0
    @inbounds for i in eachindex(bids)
        if i != winner && bids[i] > second
            second = Float64(bids[i])
        end
    end
    payments = zeros(Float64, length(bids))
    payments[winner] = second
    return AuctionOutcome(winner, (winner,), payments, winbid)
end

auction_outcome(g::SingleItemAuction, bids::AbstractVector{<:Real}) = auction_outcome(g.format, bids)

function utility(values::AbstractVector{<:Real}, outcome::AuctionOutcome)
    u = zeros(Float64, length(values))
    w = outcome.winner
    u[w] = Float64(values[w]) - outcome.payments[w]
    return u
end

end