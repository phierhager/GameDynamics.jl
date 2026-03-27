module AnalysisEvaluation

using Statistics

using ..TabularMatrixGames
using ..ApproxSolverCommon
using ..LearningDiagnostics

export TraceSnapshot
export trace_snapshot
export summarize_trace

export expected_payoff
export best_response_value
export best_response_profile_values
export unilateral_gain
export nash_gap
export epsilon_nash
export exploitability_2p_zero_sum
export exploitability_profile
export social_welfare
export coarse_correlated_gap
export correlated_gap

export current_policy_value
export average_policy_value

export trajectory_length
export reward_sum
export discounted_reward_sum
export mean_reward
export final_reward
export player_reward_sum
export player_discounted_reward_sum
export player_mean_reward
export state_visitation_counts
export action_histogram!

export cumulative_regret_value
export average_reward_value
export average_utility_value
export action_frequency_report!

# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------

@inline function _check_policy_lengths(game::TabularMatrixGames.TabularMatrixGame,
                                       x::AbstractVector,
                                       y::AbstractVector)
    length(x) == game.n_actions_p1 ||
        throw(ArgumentError("Player-1 policy length mismatch."))
    length(y) == game.n_actions_p2 ||
        throw(ArgumentError("Player-2 policy length mismatch."))
    return nothing
end

@inline function _getreward(step)
    hasproperty(step, :reward) ||
        throw(ArgumentError("Step object has no `reward` field/property."))
    return getproperty(step, :reward)
end

@inline function _getstate(step)
    hasproperty(step, :state) ||
        throw(ArgumentError("Step object has no `state` field/property."))
    return getproperty(step, :state)
end

@inline function _getaction(step)
    hasproperty(step, :action) ||
        throw(ArgumentError("Step object has no `action` field/property."))
    return getproperty(step, :action)
end

@inline function _scalar_reward(r)
    r isa Real || throw(ArgumentError(
        "Reward is not scalar. Use player-specific reward functions instead."
    ))
    return Float64(r)
end

@inline function _player_reward(r, player::Int)
    if r isa Real
        player == 1 || throw(ArgumentError("Scalar reward only supports player 1."))
        return Float64(r)
    elseif applicable(getindex, r, player)
        return Float64(r[player])
    else
        throw(ArgumentError("Reward does not support indexing for player $player."))
    end
end

# ----------------------------------------------------------------------
# Matrix-game / equilibrium evaluation
# ----------------------------------------------------------------------

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

function best_response_profile_values(game::TabularMatrixGames.TabularMatrixGame,
                                      x::AbstractVector{<:Real},
                                      y::AbstractVector{<:Real})
    br1 = best_response_value(game, 1, x, y)
    br2 = best_response_value(game, 2, x, y)
    return br1, br2
end

function unilateral_gain(game::TabularMatrixGames.TabularMatrixGame,
                         x::AbstractVector{<:Real},
                         y::AbstractVector{<:Real})
    v1, v2 = expected_payoff(game, x, y)
    br1, br2 = best_response_profile_values(game, x, y)
    return br1 - v1, br2 - v2
end

function nash_gap(game::TabularMatrixGames.TabularMatrixGame,
                  x::AbstractVector{<:Real},
                  y::AbstractVector{<:Real})
    g1, g2 = unilateral_gain(game, x, y)
    return max(g1, g2), (g1, g2)
end

function epsilon_nash(game::TabularMatrixGames.TabularMatrixGame,
                      x::AbstractVector{<:Real},
                      y::AbstractVector{<:Real})
    ε, _ = nash_gap(game, x, y)
    return ε
end

function exploitability_2p_zero_sum(game::TabularMatrixGames.TabularMatrixGame,
                                    x::AbstractVector{<:Real},
                                    y::AbstractVector{<:Real})
    g1, g2 = unilateral_gain(game, x, y)
    return g1 + g2
end

function exploitability_profile(game::TabularMatrixGames.TabularMatrixGame,
                                x::AbstractVector{<:Real},
                                y::AbstractVector{<:Real})
    return exploitability_2p_zero_sum(game, x, y)
end

function social_welfare(game::TabularMatrixGames.TabularMatrixGame,
                        x::AbstractVector{<:Real},
                        y::AbstractVector{<:Real})
    v1, v2 = expected_payoff(game, x, y)
    return v1 + v2
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
    isapprox(sπ, 1.0; atol = 1e-8) ||
        throw(ArgumentError("Joint distribution must sum to 1, got $sπ."))

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
    isapprox(sπ, 1.0; atol = 1e-8) ||
        throw(ArgumentError("Joint distribution must sum to 1, got $sπ."))

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

# ----------------------------------------------------------------------
# Solver evaluation
# ----------------------------------------------------------------------

function current_policy_value(game::TabularMatrixGames.TabularMatrixGame, ws)
    x = Vector{Float64}(undef, game.n_actions_p1)
    y = Vector{Float64}(undef, game.n_actions_p2)

    ApproxSolverCommon.current_policy!(x, ws, 1)
    ApproxSolverCommon.current_policy!(y, ws, 2)

    return expected_payoff(game, x, y)
end

function average_policy_value(game::TabularMatrixGames.TabularMatrixGame, ws)
    x = Vector{Float64}(undef, game.n_actions_p1)
    y = Vector{Float64}(undef, game.n_actions_p2)

    ApproxSolverCommon.average_policy!(x, ws, 1)
    ApproxSolverCommon.average_policy!(y, ws, 2)

    return expected_payoff(game, x, y)
end

# ----------------------------------------------------------------------
# Trajectory evaluation
# ----------------------------------------------------------------------

trajectory_length(traj) = length(traj)

function reward_sum(traj)
    acc = 0.0
    for step in traj
        acc += _scalar_reward(_getreward(step))
    end
    return acc
end

function discounted_reward_sum(traj; discount::Float64 = 1.0)
    0.0 <= discount <= 1.0 || throw(ArgumentError("discount must be in [0,1]."))
    acc = 0.0
    coeff = 1.0
    for step in traj
        acc += coeff * _scalar_reward(_getreward(step))
        coeff *= discount
    end
    return acc
end

function mean_reward(traj)
    n = length(traj)
    n == 0 && return 0.0
    return reward_sum(traj) / n
end

function final_reward(traj)
    isempty(traj) && throw(ArgumentError("Trajectory is empty."))
    return _scalar_reward(_getreward(last(traj)))
end

function player_reward_sum(traj, player::Int)
    acc = 0.0
    for step in traj
        acc += _player_reward(_getreward(step), player)
    end
    return acc
end

function player_discounted_reward_sum(traj, player::Int; discount::Float64 = 1.0)
    0.0 <= discount <= 1.0 || throw(ArgumentError("discount must be in [0,1]."))
    acc = 0.0
    coeff = 1.0
    for step in traj
        acc += coeff * _player_reward(_getreward(step), player)
        coeff *= discount
    end
    return acc
end

function player_mean_reward(traj, player::Int)
    n = length(traj)
    n == 0 && return 0.0
    return player_reward_sum(traj, player) / n
end

"""
Count visited states from a trajectory whose step objects expose `state`.
"""
function state_visitation_counts(traj)
    counts = Dict{Any,Int}()
    for step in traj
        s = _getstate(step)
        counts[s] = get(counts, s, 0) + 1
    end
    return counts
end

"""
Increment an integer action histogram from trajectory actions.

Assumes actions are integer ids in `1:length(counts)`.
"""
function action_histogram!(counts::AbstractVector{<:Integer}, traj)
    for step in traj
        a = _getaction(step)
        a isa Integer || throw(ArgumentError(
            "Encountered non-integer action $a in action_histogram!."
        ))
        1 <= a <= length(counts) || throw(BoundsError(counts, a))
        counts[a] += 1
    end
    return counts
end

# ----------------------------------------------------------------------
# Learner evaluation
# ----------------------------------------------------------------------

"""
Small named snapshot of a learner trace.
"""
Base.@kwdef struct TraceSnapshot
    rounds::Int
    cumulative_utility::Float64
    cumulative_reward::Float64
    best_fixed_utility::Float64
    cumulative_regret::Float64
    average_utility::Float64
    average_reward::Float64
end

function trace_snapshot(tr::LearningDiagnostics.LearnerTrace)
    return TraceSnapshot(
        rounds = tr.t,
        cumulative_utility = Float64(tr.cumulative_utility),
        cumulative_reward = Float64(tr.cumulative_reward),
        best_fixed_utility = Float64(tr.best_fixed_utility),
        cumulative_regret = Float64(LearningDiagnostics.cumulative_regret(tr)),
        average_utility = Float64(LearningDiagnostics.average_utility(tr)),
        average_reward = Float64(LearningDiagnostics.average_reward(tr)),
    )
end

summarize_trace(tr::LearningDiagnostics.LearnerTrace) = trace_snapshot(tr)

cumulative_regret_value(tr::LearningDiagnostics.LearnerTrace) =
    LearningDiagnostics.cumulative_regret(tr)

average_reward_value(tr::LearningDiagnostics.LearnerTrace) =
    LearningDiagnostics.average_reward(tr)

average_utility_value(tr::LearningDiagnostics.LearnerTrace) =
    LearningDiagnostics.average_utility(tr)

"""
Convert integer action counts into normalized frequencies.
"""
function action_frequency_report!(dest::AbstractVector{Float64},
                                  counts::AbstractVector{<:Integer})
    return LearningDiagnostics.empirical_action_frequencies!(dest, counts)
end

end