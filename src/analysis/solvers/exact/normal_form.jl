module ExactNormalFormSolvers

using JuMP
using HiGHS
import MathOptInterface as MOI

using ..NormalForm
using ..TabularMatrixGames

export solve_zero_sum_nash
export solve_ce
export solve_cce

export support_profiles
export profile_to_linear_index
export linear_index_to_profile

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

support_profiles(game::NormalForm.NormalFormGame{N}) where {N} =
    Tuple(NormalForm.support_profiles(game))

@inline function _strides(action_sizes::NTuple{N,Int}) where {N}
    return ntuple(i -> i == 1 ? 1 : prod(action_sizes[1:i-1]), N)
end

@inline function profile_to_linear_index(action_sizes::NTuple{N,Int}, profile::NTuple{N,Int}) where {N}
    strides = _strides(action_sizes)
    idx = 1
    @inbounds for i in 1:N
        idx += (profile[i] - 1) * strides[i]
    end
    return idx
end

@inline function linear_index_to_profile(action_sizes::NTuple{N,Int}, idx::Int) where {N}
    idx >= 1 || throw(ArgumentError("Linear index must be positive."))
    rem = idx - 1

    function build(::Val{K}, rem_) where {K}
        if K == 0
            return (), rem_
        else
            ai = action_sizes[N - K + 1]
            a = (rem_ % ai) + 1
            rest, rem2 = build(Val(K - 1), rem_ ÷ ai)
            return (a, rest...), rem2
        end
    end

    prof, _ = build(Val(N), rem)
    return prof
end

@inline function _social_welfare(game::NormalForm.NormalFormGame{N}, profile::NTuple{N,Int}) where {N}
    s = 0.0
    @inbounds for p in 1:N
        s += Float64(game.payoffs[p][profile...])
    end
    return s
end

# ----------------------------------------------------------------------
# Zero-sum Nash for 2-player normal-form games
# ----------------------------------------------------------------------

function solve_zero_sum_nash(game::NormalForm.NormalFormGame{2};
                             optimizer = HiGHS.Optimizer,
                             atol::Float64 = 1e-9)
    U = game.payoffs[1]
    m, n = size(U)

    model1 = Model(optimizer)
    set_silent(model1)

    @variable(model1, x[1:m] >= 0.0)
    @variable(model1, v)

    @constraint(model1, sum(x) == 1.0)
    @constraint(model1, [j in 1:n], sum(U[i, j] * x[i] for i in 1:m) >= v)
    @objective(model1, Max, v)

    optimize!(model1)
    termination_status(model1) == MOI.OPTIMAL ||
        error("Player 1 LP did not solve to optimality.")

    xval = value.(x)
    vval = objective_value(model1)

    model2 = Model(optimizer)
    set_silent(model2)

    @variable(model2, y[1:n] >= 0.0)
    @variable(model2, w)

    @constraint(model2, sum(y) == 1.0)
    @constraint(model2, [i in 1:m], sum(U[i, j] * y[j] for j in 1:n) <= w)
    @objective(model2, Min, w)

    optimize!(model2)
    termination_status(model2) == MOI.OPTIMAL ||
        error("Player 2 LP did not solve to optimality.")

    yval = value.(y)

    σ1 = LocalStrategies.FiniteMixedStrategy(Base.OneTo(m), xval)
    σ2 = LocalStrategies.FiniteMixedStrategy(Base.OneTo(n), yval)

    return σ1, σ2, vval
end

function solve_zero_sum_nash(game::TabularMatrixGames.TabularMatrixGame;
                             optimizer = HiGHS.Optimizer,
                             atol::Float64 = 1e-9)
    U = game.payoff_p1
    m, n = size(U)

    model1 = Model(optimizer)
    set_silent(model1)

    @variable(model1, x[1:m] >= 0.0)
    @variable(model1, v)

    @constraint(model1, sum(x) == 1.0)
    @constraint(model1, [j in 1:n], sum(U[i, j] * x[i] for i in 1:m) >= v)
    @objective(model1, Max, v)

    optimize!(model1)
    termination_status(model1) == MOI.OPTIMAL ||
        error("Player 1 LP did not solve to optimality.")

    xval = value.(x)
    vval = objective_value(model1)

    model2 = Model(optimizer)
    set_silent(model2)

    @variable(model2, y[1:n] >= 0.0)
    @variable(model2, w)

    @constraint(model2, sum(y) == 1.0)
    @constraint(model2, [i in 1:m], sum(U[i, j] * y[j] for j in 1:n) <= w)
    @objective(model2, Min, w)

    optimize!(model2)
    termination_status(model2) == MOI.OPTIMAL ||
        error("Player 2 LP did not solve to optimality.")

    yval = value.(y)
    return xval, yval, vval
end

# ----------------------------------------------------------------------
# CE / CCE for finite normal-form games
# ----------------------------------------------------------------------

function _materialized_profiles(game::NormalForm.NormalFormGame{N}) where {N}
    return Tuple(NormalForm.support_profiles(game))
end

function solve_ce(game::NormalForm.NormalFormGame{N};
                  optimizer = HiGHS.Optimizer,
                  atol::Float64 = 1e-10) where {N}
    profiles = _materialized_profiles(game)
    K = length(profiles)

    model = Model(optimizer)
    set_silent(model)

    @variable(model, π[1:K] >= 0.0)
    @constraint(model, sum(π) == 1.0)

    for p in 1:N
        A = game.action_sizes[p]
        for rec in 1:A, dev in 1:A
            rec == dev && continue
            @constraint(model,
                sum(
                    π[k] * (
                        Float64(game.payoffs[p][profiles[k]...]) -
                        Float64(game.payoffs[p][Base.setindex(profiles[k], dev, p)...])
                    )
                    for k in 1:K if profiles[k][p] == rec
                ) >= 0.0
            )
        end
    end

    @objective(model, Max, sum(π[k] * _social_welfare(game, profiles[k]) for k in 1:K))

    optimize!(model)
    termination_status(model) == MOI.OPTIMAL ||
        error("CE LP did not solve to optimality.")

    probs = value.(π)
    keep = Tuple(k for k in 1:K if probs[k] > atol)
    prof_support = ntuple(i -> profiles[keep[i]], length(keep))
    prob_support = ntuple(i -> probs[keep[i]], length(keep))

    return JointStrategies.CorrelatedRecommendationDevice(prof_support, prob_support)
end

function solve_cce(game::NormalForm.NormalFormGame{N};
                   optimizer = HiGHS.Optimizer,
                   atol::Float64 = 1e-10) where {N}
    profiles = _materialized_profiles(game)
    K = length(profiles)

    model = Model(optimizer)
    set_silent(model)

    @variable(model, π[1:K] >= 0.0)
    @constraint(model, sum(π) == 1.0)

    for p in 1:N
        A = game.action_sizes[p]
        for dev in 1:A
            @constraint(model,
                sum(
                    π[k] * (
                        Float64(game.payoffs[p][profiles[k]...]) -
                        Float64(game.payoffs[p][Base.setindex(profiles[k], dev, p)...])
                    )
                    for k in 1:K
                ) >= 0.0
            )
        end
    end

    @objective(model, Max, sum(π[k] * _social_welfare(game, profiles[k]) for k in 1:K))

    optimize!(model)
    termination_status(model) == MOI.OPTIMAL ||
        error("CCE LP did not solve to optimality.")

    probs = value.(π)
    keep = Tuple(k for k in 1:K if probs[k] > atol)
    prof_support = ntuple(i -> profiles[keep[i]], length(keep))
    prob_support = ntuple(i -> probs[keep[i]], length(keep))

    return JointStrategies.CorrelatedRecommendationDevice(prof_support, prob_support)
end

end