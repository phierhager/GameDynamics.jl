module NormalForm

using Random
using ..Kernel
using ..Spaces
using ..Capabilities
using ..Exact
using ..Spec
using ..Strategies
using ..Classification

export NormalFormGame, NormalFormState
export pure_payoff, expected_payoff
export best_response_values, best_response
export payoff_tensor, action_count, action_counts
export support_profiles

struct NormalFormState <: Kernel.AbstractState
    played::Bool
end

struct NormalFormGame{N,T<:Tuple,R} <: Kernel.AbstractFixedGame{N,R}
    payoffs::T
    action_sizes::NTuple{N,Int}
    spec::Spec.GameSpec
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

    R = NTuple{N,Float64}
    spec = Spec.GameSpec(
        perfect_information = true,
        perfect_recall = true,
        stochastic = false,
        simultaneous_moves = true,
        zero_sum = N == 2 ? all(payoffs[1] .+ payoffs[2] .== 0) : false,
        general_sum = true,
        horizon_kind = Spec.EPISODIC,
        player_model = Spec.FIXED_PLAYERS,
        max_steps = 1,
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

Capabilities.has_action_mask(::Type{<:NormalFormGame}) = Val(true)
Capabilities.has_state_space(::Type{<:NormalFormGame}) = Val(true)
Capabilities.has_action_space(::Type{<:NormalFormGame}) = Val(true)

Spec.game_spec(g::NormalFormGame) = g.spec
Classification.is_normal_form(::NormalFormGame) = true

Kernel.init_state(::NormalFormGame, rng::AbstractRNG = Random.default_rng()) = NormalFormState(false)

Kernel.node_kind(::NormalFormGame, s::NormalFormState) =
    s.played ? Kernel.TERMINAL : Kernel.SIMULTANEOUS

Kernel.active_players(::NormalFormGame{N}, s::NormalFormState) where {N} = Base.OneTo(N)

Kernel.legal_actions(g::NormalFormGame, s::NormalFormState, player::Int) =
    Base.OneTo(g.action_sizes[player])

Kernel.num_base_actions(g::NormalFormGame, player::Int) = g.action_sizes[player]

Kernel.legal_action_mask(g::NormalFormGame, s::NormalFormState, player::Int) =
    ntuple(_ -> true, g.action_sizes[player])

Kernel.encode_action(::NormalFormGame, player::Int, action::Int) = action
Kernel.decode_action(::NormalFormGame, player::Int, index::Int) = index

function Kernel.step(g::NormalFormGame{N}, s::NormalFormState, a::Kernel.JointAction{N}, rng::AbstractRNG = Random.default_rng()) where {N}
    s.played && throw(ArgumentError("Cannot step from a terminal normal-form state."))
    profile = a.actions
    @inbounds for p in 1:N
        1 <= profile[p] <= g.action_sizes[p] ||
            throw(ArgumentError("Illegal action $(profile[p]) for player $p."))
    end
    rewards = ntuple(p -> Float64(g.payoffs[p][profile...]), N)
    return NormalFormState(true), rewards, true
end

Kernel.observe(::NormalFormGame, s::NormalFormState, player::Int) = nothing

Exact.state_space(::NormalFormGame) =
    Spaces.FiniteSpace((NormalFormState(false), NormalFormState(true)))

Exact.action_space(g::NormalFormGame, s::NormalFormState, player::Int) =
    Spaces.IndexedDiscreteSpace(g.action_sizes[player])

function Exact.terminal_payoffs(::NormalFormGame, s::NormalFormState)
    s.played || throw(ArgumentError("terminal_payoffs is only valid on terminal states."))
    error("NormalFormGame uses transition rewards on play; terminal_payoffs is not a meaningful state-utility API here.")
end

pure_payoff(g::NormalFormGame{N}, profile::NTuple{N,Int}) where {N} =
    ntuple(p -> Float64(g.payoffs[p][profile...]), N)

function expected_payoff(g::NormalFormGame{2},
                         profile::Tuple{Strategies.FiniteMixedStrategy,Strategies.FiniteMixedStrategy})
    s1, s2 = profile
    A1, P1 = Strategies.support(s1), Strategies.probabilities(s1)
    A2, P2 = Strategies.support(s2), Strategies.probabilities(s2)

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
                         profile::Tuple{Vararg{Strategies.FiniteMixedStrategy,N}}) where {N}
    supports = ntuple(i -> Strategies.support(profile[i]), N)
    probs = ntuple(i -> Strategies.probabilities(profile[i]), N)
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

function expected_payoff(g::NormalFormGame{N}, corr::Strategies.CorrelatedStrategy) where {N}
    acc = zeros(Float64, N)
    S = Strategies.support(corr)
    P = Strategies.probabilities(corr)

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
                              profile::Tuple{Strategies.FiniteMixedStrategy,Strategies.FiniteMixedStrategy})
    1 <= player <= 2 || throw(ArgumentError("Invalid player index $player."))
    vals = zeros(Float64, g.action_sizes[player])

    if player == 1
        A2 = Strategies.support(profile[2])
        P2 = Strategies.probabilities(profile[2])
        @inbounds for a1 in 1:g.action_sizes[1]
            acc = 0.0
            for j in eachindex(P2)
                acc += P2[j] * Float64(g.payoffs[1][a1, A2[j]])
            end
            vals[a1] = acc
        end
    else
        A1 = Strategies.support(profile[1])
        P1 = Strategies.probabilities(profile[1])
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
                              profile::Tuple{Vararg{Strategies.FiniteMixedStrategy,N}}) where {N}
    1 <= player <= N || throw(ArgumentError("Invalid player index $player."))

    opp_supports = ntuple(i -> Strategies.support(profile[i]), N)
    opp_probs = ntuple(i -> Strategies.probabilities(profile[i]), N)
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
                       profile::Tuple{Vararg{Strategies.FiniteMixedStrategy,N}}) where {N}
    vals = best_response_values(g, player, profile)
    best_idx = argmax(vals)
    return best_idx, vals[best_idx]
end

end