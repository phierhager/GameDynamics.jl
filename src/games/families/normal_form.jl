module NormalForm

using Random
using ..Kernel
using ..Spaces
using ..Exact
using ..Spec
using ..Classification
using ..DecisionRulesInterface
using ..DirectDecisionRules
using ..JointDecisionRules

export NormalFormGame, NormalFormState
export pure_payoff, expected_payoff
export best_response_values, best_response
export payoff_tensor, action_count, action_counts
export support_profiles

struct NormalFormState <: Kernel.AbstractState
    played::Bool
end

struct NormalFormGame{N,T<:Tuple,R} <: Kernel.AbstractGame{N,R}
    payoffs::T
    action_sizes::NTuple{N,Int}
    spec::Spec.GameSpec
end

function _infer_payoff_kind(payoffs::T) where {T<:Tuple}
    N = length(payoffs)

    if N == 2
        zs = true
        @inbounds for I in eachindex(payoffs[1])
            if !isapprox(payoffs[1][I] + payoffs[2][I], 0.0; atol=1e-12, rtol=1e-10)
                zs = false
                break
            end
        end
        zs && return Spec.ZERO_SUM
    end

    total = payoffs[1]
    for p in 2:N
        total = total .+ payoffs[p]
    end

    first_val = Float64(total[first(eachindex(total))])
    constant_sum = true
    @inbounds for x in total
        if !isapprox(Float64(x), first_val; atol=1e-12, rtol=1e-10)
            constant_sum = false
            break
        end
    end

    return constant_sum ? Spec.CONSTANT_SUM : Spec.GENERAL_SUM
end

function _infer_reward_sharing(payoff_kind::Spec.PayoffKind)
    if payoff_kind == Spec.ZERO_SUM || payoff_kind == Spec.CONSTANT_SUM || payoff_kind == Spec.GENERAL_SUM
        return Spec.INDEPENDENT_REWARD
    else
        return Spec.UNKNOWN_REWARD_SHARING
    end
end

function NormalFormGame(payoffs::T) where {T<:Tuple}
    N = length(payoffs)
    N > 0 || throw(ArgumentError("NormalFormGame requires at least one player."))

    dims = size(payoffs[1])
    length(dims) == N ||
        throw(ArgumentError("Each payoff tensor must have one axis per player."))

    for p in 2:N
        size(payoffs[p]) == dims ||
            throw(ArgumentError("All payoff tensors must have identical shape."))
    end

    payoff_kind = _infer_payoff_kind(payoffs)
    R = NTuple{N,Float64}

    spec = Spec.GameSpec(
        horizon_kind = Spec.EPISODIC,
        payoff_kind = payoff_kind,
        max_steps = 1,
        default_discount = 1.0,
        perfect_information = false,
        stochastic = false,
        simultaneous_moves = true,
        observation_kind = Spec.UNKNOWN_OBSERVATION,
        cooperative = false,
        reward_sharing = _infer_reward_sharing(payoff_kind),
    )

    return NormalFormGame{N,T,R}(payoffs, ntuple(i -> dims[i], N), spec)
end

payoff_tensor(g::NormalFormGame, player::Int) = g.payoffs[player]
action_count(g::NormalFormGame, player::Int) = g.action_sizes[player]
action_counts(g::NormalFormGame{N}) where {N} = g.action_sizes

function support_profiles(g::NormalFormGame{N}) where {N}
    ranges = ntuple(i -> Base.OneTo(g.action_sizes[i]), N)
    return Iterators.product(ranges...)
end

Kernel.action_mode(::Type{<:NormalFormGame}) = Kernel.IndexedActions
Kernel.has_action_mask(::Type{<:NormalFormGame}) = true

Spec.game_spec(g::NormalFormGame) = g.spec
Classification.is_normal_form(::NormalFormGame) = true
Classification.is_extensive_form(::NormalFormGame) = false

Kernel.init_state(::NormalFormGame, rng::AbstractRNG = Random.default_rng()) = NormalFormState(false)

Kernel.node_kind(::NormalFormGame, s::NormalFormState) =
    s.played ? Kernel.TERMINAL : Kernel.SIMULTANEOUS

Kernel.active_players(::NormalFormGame{N}, s::NormalFormState) where {N} = Base.OneTo(N)

Kernel.legal_actions(g::NormalFormGame, s::NormalFormState, player::Int) =
    Base.OneTo(g.action_sizes[player])

Kernel.indexed_action_count(g::NormalFormGame, player::Int) = g.action_sizes[player]

Kernel.legal_action_mask(g::NormalFormGame, s::NormalFormState, player::Int) =
    ntuple(_ -> true, g.action_sizes[player])

function Kernel.step(g::NormalFormGame{N},
                     s::NormalFormState,
                     a::Kernel.JointAction,
                     rng::AbstractRNG = Random.default_rng()) where {N}
    s.played && throw(ArgumentError("Cannot step from a terminal normal-form state."))

    Kernel.validate_joint_action(g, s, a)

    profile = Tuple(a)
    length(profile) == N || throw(ArgumentError(
        "NormalFormGame expected $N simultaneous actions, got $(length(profile))."
    ))

    rewards = ntuple(p -> Float64(g.payoffs[p][profile...]), N)
    return NormalFormState(true), rewards
end

Kernel.observe(::NormalFormGame, s::NormalFormState, player::Int) = nothing

pure_payoff(g::NormalFormGame{N}, profile::NTuple{N,Int}) where {N} =
    ntuple(p -> Float64(g.payoffs[p][profile...]), N)

function expected_payoff(g::NormalFormGame{2},
                         profile::Tuple{DirectDecisionRules.FiniteMixedDecisionRule,DirectDecisionRules.FiniteMixedDecisionRule})
    s1, s2 = profile
    A1, P1 = DecisionRulesInterface.support(s1), DecisionRulesInterface.probabilities(s1)
    A2, P2 = DecisionRulesInterface.support(s2), DecisionRulesInterface.probabilities(s2)

    acc1 = 0.0
    acc2 = 0.0
    @inbounds for i in eachindex(P1)
        a1 = A1[i]
        p1 = P1[i]
        for j in eachindex(P2)
            m = p1 * P2[j]
            a2 = A2[j]
            acc1 += m * Float64(g.payoffs[1][a1, a2])
            acc2 += m * Float64(g.payoffs[2][a1, a2])
        end
    end
    return (acc1, acc2)
end

function expected_payoff(g::NormalFormGame{N},
                         profile::Tuple{Vararg{DirectDecisionRules.FiniteMixedDecisionRule,N}}) where {N}
    supports = ntuple(i -> DecisionRulesInterface.support(profile[i]), N)
    probs = ntuple(i -> DecisionRulesInterface.probabilities(profile[i]), N)
    ranges = ntuple(i -> Base.OneTo(length(probs[i])), N)

    acc = zeros(Float64, N)
    @inbounds for idxs in Iterators.product(ranges...)
        joint = ntuple(i -> supports[i][idxs[i]], N)
        mass = 1.0
        for i in 1:N
            mass *= probs[i][idxs[i]]
        end
        for p in 1:N
            acc[p] += mass * Float64(g.payoffs[p][joint...])
        end
    end
    return ntuple(i -> acc[i], N)
end

function expected_payoff(g::NormalFormGame{N}, corr::JointDecisionRules.CorrelatedActionRule) where {N}
    acc = zeros(Float64, N)
    S = DecisionRulesInterface.support(corr)
    P = DecisionRulesInterface.probabilities(corr)

    @inbounds for i in eachindex(P)
        profile = S[i]
        mass = P[i]
        for p in 1:N
            acc[p] += mass * Float64(g.payoffs[p][profile...])
        end
    end
    return ntuple(i -> acc[i], N)
end

function best_response_values(g::NormalFormGame{2},
                              player::Int,
                              profile::Tuple{DirectDecisionRules.FiniteMixedDecisionRule,DirectDecisionRules.FiniteMixedDecisionRule})
    1 <= player <= 2 || throw(ArgumentError("Invalid player index $player."))
    vals = zeros(Float64, g.action_sizes[player])

    if player == 1
        A2 = DecisionRulesInterface.support(profile[2])
        P2 = DecisionRulesInterface.probabilities(profile[2])
        @inbounds for a1 in 1:g.action_sizes[1]
            acc = 0.0
            for j in eachindex(P2)
                acc += P2[j] * Float64(g.payoffs[1][a1, A2[j]])
            end
            vals[a1] = acc
        end
    else
        A1 = DecisionRulesInterface.support(profile[1])
        P1 = DecisionRulesInterface.probabilities(profile[1])
        @inbounds for a2 in 1:g.action_sizes[2]
            acc = 0.0
            for i in eachindex(P1)
                acc += P1[i] * Float64(g.payoffs[2][A1[i], a2])
            end
            vals[a2] = acc
        end
    end

    return vals
end

function best_response_values(g::NormalFormGame{N},
                              player::Int,
                              profile::Tuple{Vararg{DirectDecisionRules.FiniteMixedDecisionRule,N}}) where {N}
    1 <= player <= N || throw(ArgumentError("Invalid player index $player."))

    opp_supports = ntuple(i -> DecisionRulesInterface.support(profile[i]), N)
    opp_probs = ntuple(i -> DecisionRulesInterface.probabilities(profile[i]), N)
    values = zeros(Float64, g.action_sizes[player])

    function recurse_build(action_i, pidx, current_profile, mass)
        if pidx > N
            values[action_i] += mass * Float64(g.payoffs[player][current_profile...])
            return
        end

        if pidx == player
            recurse_build(action_i, pidx + 1, Base.setindex(current_profile, action_i, pidx), mass)
        else
            S = opp_supports[pidx]
            P = opp_probs[pidx]
            @inbounds for j in eachindex(P)
                recurse_build(action_i, pidx + 1, Base.setindex(current_profile, S[j], pidx), mass * P[j])
            end
        end
    end

    base_profile = ntuple(_ -> 1, N)
    for a in 1:g.action_sizes[player]
        recurse_build(a, 1, base_profile, 1.0)
    end

    return values
end

function best_response(g::NormalFormGame{N},
                       player::Int,
                       profile::Tuple{Vararg{DirectDecisionRules.FiniteMixedDecisionRule,N}}) where {N}
    vals = best_response_values(g, player, profile)
    best_idx = argmax(vals)
    return best_idx, vals[best_idx]
end

end