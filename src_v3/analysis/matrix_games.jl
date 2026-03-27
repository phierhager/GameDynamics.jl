module MatrixGameAnalysis

using ..TabularMatrixGames

export expected_payoff
export best_response_value
export nash_gap
export exploitability_2p_zero_sum

@inline function _check_policy_lengths(game::TabularMatrixGames.TabularMatrixGame,
                                       x::AbstractVector,
                                       y::AbstractVector)
    length(x) == game.n_actions_p1 ||
        throw(ArgumentError("Player-1 policy length mismatch."))
    length(y) == game.n_actions_p2 ||
        throw(ArgumentError("Player-2 policy length mismatch."))
    return nothing
end

function expected_payoff(game::TabularMatrixGames.TabularMatrixGame,
                         x::AbstractVector{<:Real},
                         y::AbstractVector{<:Real})
    _check_policy_lengths(game, x, y)

    U1 = game.payoff_p1
    U2 = game.payoff_p2

    v1 = 0.0
    v2 = 0.0
    @inbounds for i in 1:game.n_actions_p1
        xi = x[i]
        for j in 1:game.n_actions_p2
            m = xi * y[j]
            v1 += m * U1[i, j]
            v2 += m * U2[i, j]
        end
    end
    return v1, v2
end

function best_response_value(game::TabularMatrixGames.TabularMatrixGame,
                             player::Int,
                             x::AbstractVector{<:Real},
                             y::AbstractVector{<:Real})
    _check_policy_lengths(game, x, y)

    if player == 1
        U1 = game.payoff_p1
        best = -Inf
        @inbounds for i in 1:game.n_actions_p1
            acc = 0.0
            for j in 1:game.n_actions_p2
                acc += U1[i, j] * y[j]
            end
            best = max(best, acc)
        end
        return best

    elseif player == 2
        U2 = game.payoff_p2
        best = -Inf
        @inbounds for j in 1:game.n_actions_p2
            acc = 0.0
            for i in 1:game.n_actions_p1
                acc += U2[i, j] * x[i]
            end
            best = max(best, acc)
        end
        return best

    else
        throw(ArgumentError("Only players 1 and 2 are supported."))
    end
end

function nash_gap(game::TabularMatrixGames.TabularMatrixGame,
                  x::AbstractVector{<:Real},
                  y::AbstractVector{<:Real})
    v = expected_payoff(game, x, y)
    br1 = best_response_value(game, 1, x, y)
    br2 = best_response_value(game, 2, x, y)
    g1 = br1 - v[1]
    g2 = br2 - v[2]
    return max(g1, g2), (g1, g2)
end

function exploitability_2p_zero_sum(game::TabularMatrixGames.TabularMatrixGame,
                                    x::AbstractVector{<:Real},
                                    y::AbstractVector{<:Real})
    v1, v2 = expected_payoff(game, x, y)
    br1 = best_response_value(game, 1, x, y)
    br2 = best_response_value(game, 2, x, y)
    return (br1 - v1) + (br2 - v2)
end

end