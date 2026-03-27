module EquilibriumMetrics

using ..TabularMatrixGames
using ..MatrixGameAnalysis

export expected_payoff
export social_welfare
export unilateral_gain
export nash_gap
export epsilon_nash
export coarse_correlated_gap
export correlated_gap

@inline function _check_policy_lengths(game::TabularMatrixGames.TabularMatrixGame,
                                       x::AbstractVector,
                                       y::AbstractVector)
    length(x) == game.n_actions_p1 ||
        throw(ArgumentError("Player-1 policy length mismatch."))
    length(y) == game.n_actions_p2 ||
        throw(ArgumentError("Player-2 policy length mismatch."))
    return nothing
end

expected_payoff(game::TabularMatrixGames.TabularMatrixGame,
                x::AbstractVector{<:Real},
                y::AbstractVector{<:Real}) =
    MatrixGameAnalysis.expected_payoff(game, x, y)

function social_welfare(game::TabularMatrixGames.TabularMatrixGame,
                        x::AbstractVector{<:Real},
                        y::AbstractVector{<:Real})
    v1, v2 = expected_payoff(game, x, y)
    return v1 + v2
end

function unilateral_gain(game::TabularMatrixGames.TabularMatrixGame,
                         x::AbstractVector{<:Real},
                         y::AbstractVector{<:Real})
    v1, v2 = expected_payoff(game, x, y)
    br1 = MatrixGameAnalysis.best_response_value(game, 1, x, y)
    br2 = MatrixGameAnalysis.best_response_value(game, 2, x, y)
    return br1 - v1, br2 - v2
end

nash_gap(game::TabularMatrixGames.TabularMatrixGame,
         x::AbstractVector{<:Real},
         y::AbstractVector{<:Real}) =
    MatrixGameAnalysis.nash_gap(game, x, y)

"""
Return the scalar epsilon such that the profile is an `ε`-Nash equilibrium.
"""
function epsilon_nash(game::TabularMatrixGames.TabularMatrixGame,
                      x::AbstractVector{<:Real},
                      y::AbstractVector{<:Real})
    ε, _ = nash_gap(game, x, y)
    return ε
end

"""
Maximum coarse-correlated-equilibrium deviation violation for a joint distribution `π`
over action profiles of a 2-player matrix game.

`π` must be an `m x n` matrix whose entries are nonnegative and sum to one.
"""
function coarse_correlated_gap(game::TabularMatrixGames.TabularMatrixGame,
                               π::AbstractMatrix{<:Real})
    m, n = size(π)
    m == game.n_actions_p1 || throw(ArgumentError("Joint distribution row count mismatch."))
    n == game.n_actions_p2 || throw(ArgumentError("Joint distribution column count mismatch."))

    sπ = sum(π)
    isapprox(sπ, 1.0; atol = 1e-8) || throw(ArgumentError("Joint distribution must sum to 1, got $sπ."))

    U1 = game.payoff_p1
    U2 = game.payoff_p2

    v1 = 0.0
    v2 = 0.0
    @inbounds for i in 1:m, j in 1:n
        p = Float64(π[i, j])
        v1 += p * U1[i, j]
        v2 += p * U2[i, j]
    end

    best1 = -Inf
    @inbounds for a1 in 1:m
        acc = 0.0
        for i in 1:m, j in 1:n
            acc += Float64(π[i, j]) * U1[a1, j]
        end
        best1 = max(best1, acc)
    end

    best2 = -Inf
    @inbounds for a2 in 1:n
        acc = 0.0
        for i in 1:m, j in 1:n
            acc += Float64(π[i, j]) * U2[i, a2]
        end
        best2 = max(best2, acc)
    end

    g1 = best1 - v1
    g2 = best2 - v2
    return max(g1, g2), (g1, g2)
end

"""
Maximum correlated-equilibrium deviation violation for a joint distribution `π`
over action profiles of a 2-player matrix game.

`π` must be an `m x n` matrix whose entries are nonnegative and sum to one.
"""
function correlated_gap(game::TabularMatrixGames.TabularMatrixGame,
                        π::AbstractMatrix{<:Real})
    m, n = size(π)
    m == game.n_actions_p1 || throw(ArgumentError("Joint distribution row count mismatch."))
    n == game.n_actions_p2 || throw(ArgumentError("Joint distribution column count mismatch."))

    sπ = sum(π)
    isapprox(sπ, 1.0; atol = 1e-8) || throw(ArgumentError("Joint distribution must sum to 1, got $sπ."))

    U1 = game.payoff_p1
    U2 = game.payoff_p2

    max_gap = 0.0
    gaps = Float64[]

    @inbounds for rec in 1:m, dev in 1:m
        rec == dev && continue
        acc = 0.0
        for j in 1:n
            acc += Float64(π[rec, j]) * (U1[dev, j] - U1[rec, j])
        end
        push!(gaps, acc)
        max_gap = max(max_gap, acc)
    end

    @inbounds for rec in 1:n, dev in 1:n
        rec == dev && continue
        acc = 0.0
        for i in 1:m
            acc += Float64(π[i, rec]) * (U2[i, dev] - U2[i, rec])
        end
        push!(gaps, acc)
        max_gap = max(max_gap, acc)
    end

    return max_gap, Tuple(gaps)
end

end