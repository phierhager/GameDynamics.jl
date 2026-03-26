module Stackelberg

using ..NormalForm
using ..Strategies
using ..Classification

export StackelbergGame
export leader, follower
export follower_best_response, leader_value
export ARBITRARY_TIE_BREAK, FAVOR_LEADER_TIE_BREAK, FAVOR_FOLLOWER_TIE_BREAK

const ARBITRARY_TIE_BREAK = :arbitrary
const FAVOR_LEADER_TIE_BREAK = :favor_leader
const FAVOR_FOLLOWER_TIE_BREAK = :favor_follower

@inline function _argmax_all(vals::AbstractVector; atol=1e-12, rtol=1e-10)
    isempty(vals) && throw(ArgumentError("vals must be nonempty."))
    m = maximum(vals)
    return findall(v -> isapprox(v, m; atol=atol, rtol=rtol), vals)
end

"""
Two-player Stackelberg wrapper over a base normal-form game.

This is a theory-level semantic object:
- `leader` commits first
- `follower` best-responds subject to tie-breaking convention
"""
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

@inline leader(g::StackelbergGame) = g.leader
@inline follower(g::StackelbergGame) = g.follower

@inline function _degenerate_mixed(action::Integer)
    return Strategies.FiniteMixedStrategy((action,), (1.0,))
end

function _validate_strategy_domain(s::Strategies.FiniteMixedStrategy, n::Int, who::AbstractString)
    A = Strategies.support(s)
    isempty(A) && throw(ArgumentError("$who strategy must have nonempty support."))

    @inbounds for i in eachindex(A)
        a = A[i]
        a isa Integer && 1 <= a <= n ||
            throw(ArgumentError("$who strategy contains illegal action $a; expected actions in 1:$n."))
    end

    return s
end

"""
Compute a follower best response to a leader mixed strategy.

Returns `(a_f, v_f)` where:
- `a_f` is a pure follower best-response action
- `v_f` is the follower best-response value

Tie-breaking policies:
- `ARBITRARY_TIE_BREAK`
- `FAVOR_LEADER_TIE_BREAK`
- `FAVOR_FOLLOWER_TIE_BREAK`

For follower payoff, all exact best responses are equivalent up to the tolerance
used in `_argmax_all`, so `FAVOR_FOLLOWER_TIE_BREAK` currently selects the first
best-response representative.
"""
function follower_best_response(g::StackelbergGame,
                                leader_strategy::Strategies.FiniteMixedStrategy;
                                tie_break::Symbol = ARBITRARY_TIE_BREAK)
    base = g.base_game
    li = g.leader
    fi = g.follower

    _validate_strategy_domain(leader_strategy, base.action_sizes[li], "Leader")

    prof = if li == 1
        (leader_strategy, Strategies.FiniteMixedStrategy(ones(base.action_sizes[2])))
    else
        (Strategies.FiniteMixedStrategy(ones(base.action_sizes[1])), leader_strategy)
    end

    vals = NormalForm.best_response_values(base, fi, prof)
    best = _argmax_all(vals)

    a = if tie_break === ARBITRARY_TIE_BREAK
        first(best)

    elseif tie_break === FAVOR_FOLLOWER_TIE_BREAK
        # All best responses are payoff-equivalent for the follower under the
        # current best-response tolerance, so any representative is acceptable.
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

"""
Evaluate the leader payoff induced by a leader mixed strategy under follower
best-response behavior and the given tie-breaking convention.
"""
function leader_value(g::StackelbergGame,
                      leader_strategy::Strategies.FiniteMixedStrategy;
                      tie_break::Symbol = ARBITRARY_TIE_BREAK)
    _validate_strategy_domain(leader_strategy, g.base_game.action_sizes[g.leader], "Leader")

    a_f, _ = follower_best_response(g, leader_strategy; tie_break=tie_break)
    follower_strategy = _degenerate_mixed(a_f)

    prof = if g.leader == 1
        (leader_strategy, follower_strategy)
    else
        (follower_strategy, leader_strategy)
    end

    return NormalForm.expected_payoff(g.base_game, prof)[g.leader]
end

Classification.is_stackelberg_game(::StackelbergGame) = true
Classification.is_hierarchical_game(::StackelbergGame) = true

end