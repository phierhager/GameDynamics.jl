module StackelbergGames

using ..NormalForm
using ..LocalStrategies
using ..StrategyInterface

export StackelbergGame
export leader
export follower
export follower_best_response
export leader_value
export ARBITRARY_TIE_BREAK
export FAVOR_LEADER_TIE_BREAK

const ARBITRARY_TIE_BREAK = :arbitrary
const FAVOR_LEADER_TIE_BREAK = :favor_leader

struct StackelbergGame{G<:NormalForm.NormalFormGame{2}}
    base_game::G
    leader::Int
    follower::Int

    function StackelbergGame(base_game::G, leader::Int, follower::Int) where {G<:NormalForm.NormalFormGame{2}}
        leader in (1, 2) || throw(ArgumentError("leader must be 1 or 2."))
        follower in (1, 2) || throw(ArgumentError("follower must be 1 or 2."))
        leader != follower || throw(ArgumentError("Leader and follower must differ."))
        new{G}(base_game, leader, follower)
    end
end

leader(g::StackelbergGame) = g.leader
follower(g::StackelbergGame) = g.follower

@inline function _argmax_all(vals::AbstractVector; atol=1e-12, rtol=1e-10)
    isempty(vals) && throw(ArgumentError("vals must be nonempty."))
    m = maximum(vals)
    return findall(v -> isapprox(v, m; atol=atol, rtol=rtol), vals)
end

@inline _degenerate_mixed(action::Integer) =
    LocalStrategies.FiniteMixedStrategy((action,), (1.0,))

function _validate_strategy_domain(s::LocalStrategies.FiniteMixedStrategy, n::Int, who::AbstractString)
    A = StrategyInterface.support(s)
    isempty(A) && throw(ArgumentError("$who strategy must have nonempty support."))
    @inbounds for i in eachindex(A)
        a = A[i]
        a isa Integer && 1 <= a <= n ||
            throw(ArgumentError("$who strategy contains illegal action $a; expected actions in 1:$n."))
    end
    return s
end

function follower_best_response(g::StackelbergGame,
                                leader_strategy::LocalStrategies.FiniteMixedStrategy;
                                tie_break::Symbol = ARBITRARY_TIE_BREAK)
    base = g.base_game
    li = g.leader
    fi = g.follower

    _validate_strategy_domain(leader_strategy, base.action_sizes[li], "Leader")

    prof = if li == 1
        (leader_strategy, LocalStrategies.FiniteMixedStrategy(ones(base.action_sizes[2])))
    else
        (LocalStrategies.FiniteMixedStrategy(ones(base.action_sizes[1])), leader_strategy)
    end

    vals = NormalForm.best_response_values(base, fi, prof)
    best = _argmax_all(vals)

    a = if tie_break === ARBITRARY_TIE_BREAK
        first(best)
    elseif tie_break === FAVOR_LEADER_TIE_BREAK
        leader_vals = Vector{Float64}(undef, length(best))
        @inbounds for i in eachindex(best)
            af = best[i]
            follower_strategy = _degenerate_mixed(af)
            full_prof = if li == 1
                (leader_strategy, follower_strategy)
            else
                (follower_strategy, leader_strategy)
            end
            leader_vals[i] = NormalForm.expected_payoff(base, full_prof)[li]
        end
        best[argmax(leader_vals)]
    else
        throw(ArgumentError("Unknown tie_break policy: $tie_break"))
    end

    return a, vals[a]
end

function leader_value(g::StackelbergGame,
                      leader_strategy::LocalStrategies.FiniteMixedStrategy;
                      tie_break::Symbol = ARBITRARY_TIE_BREAK)
    _validate_strategy_domain(leader_strategy, g.base_game.action_sizes[g.leader], "Leader")

    a_f, _ = follower_best_response(g, leader_strategy; tie_break = tie_break)
    follower_strategy = _degenerate_mixed(a_f)

    prof = if g.leader == 1
        (leader_strategy, follower_strategy)
    else
        (follower_strategy, leader_strategy)
    end

    return NormalForm.expected_payoff(g.base_game, prof)[g.leader]
end

end