module CompiledMarkovModels

using ..Compiled
using ..Kernel
using ..Exact
using ..Encodings

export CompiledMDP
export CompiledZeroSumMarkovGame
export compile_mdp
export compile_zero_sum_markov_game
export n_states, n_actions

struct CompiledMDP{SE,S} <: Compiled.AbstractCompiledMarkovModel
    n_states::Int
    n_actions::Vector{Int}
    action_ptr::Vector{Int}      # length n_states + 1
    trans_ptr::Vector{Int}       # length total_state_actions + 1
    next_state::Vector{Int}
    prob::Vector{Float64}
    reward::Vector{Float64}
    state_encoder::SE
    states::Vector{S}
end

struct CompiledZeroSumMarkovGame{SE,S,A1T,A2T} <: Compiled.AbstractCompiledMarkovModel
    n_states::Int
    n_actions_p1::Vector{Int}
    n_actions_p2::Vector{Int}
    pair_ptr::Vector{Int}        # length n_states + 1
    trans_ptr::Vector{Int}       # length total_joint_actions + 1
    next_state::Vector{Int}
    prob::Vector{Float64}
    reward::Vector{Float64}      # player-1 reward
    state_encoder::SE
    states::Vector{S}
    legal_actions_p1::Vector{A1T}
    legal_actions_p2::Vector{A2T}
end

n_states(m::CompiledMDP) = m.n_states
n_states(m::CompiledZeroSumMarkovGame) = m.n_states
n_actions(m::CompiledMDP, s::Int) = m.n_actions[s]
n_actions(m::CompiledZeroSumMarkovGame, s::Int) = (m.n_actions_p1[s], m.n_actions_p2[s])

@inline _scalar_reward(r) = r isa Real ? Float64(r) : Float64(r[1])

@inline _action_count(xs::Base.OneTo) = length(xs)
@inline _action_count(xs::AbstractVector) = length(xs)
@inline _action_count(xs::Tuple) = length(xs)
@inline _action_count(xs) = length(collect(xs))

@inline _materialize_actions(xs::Base.OneTo) = xs
@inline _materialize_actions(xs::AbstractVector) = xs
@inline _materialize_actions(xs::Tuple) = xs
@inline _materialize_actions(xs) = collect(xs)

function _build_state_encoder(states::AbstractVector{S}) where {S}
    enc = Encodings.DenseEncoder{S}()
    Encodings.sizehint!(enc, length(states))
    @inbounds for s in states
        Encodings.encode!(enc, s)
    end
    return enc
end

function compile_mdp(game::Kernel.AbstractGame, states)
    Kernel.num_players(game) == 1 ||
        throw(ArgumentError("compile_mdp requires a single-player game."))

    state_vec = collect(states)
    S = eltype(state_vec)
    enc = _build_state_encoder(state_vec)
    nS = length(state_vec)

    n_actions_vec = Vector{Int}(undef, nS)
    action_ptr = Vector{Int}(undef, nS + 1)

    total_sa = 0
    @inbounds for sidx in 1:nS
        s = state_vec[sidx]
        nk = Kernel.node_kind(game, s)
        nA = if nk == Kernel.TERMINAL || nk == Kernel.CHANCE
            0
        else
            _action_count(Kernel.legal_actions(game, s, 1))
        end
        n_actions_vec[sidx] = nA
        action_ptr[sidx] = total_sa + 1
        total_sa += nA
    end
    action_ptr[end] = total_sa + 1

    trans_ptr = Vector{Int}(undef, total_sa + 1)
    next_state = Int[]
    prob = Float64[]
    reward = Float64[]
    sizehint!(next_state, max(total_sa, 16))
    sizehint!(prob, max(total_sa, 16))
    sizehint!(reward, max(total_sa, 16))

    sa_idx = 1
    @inbounds for sidx in 1:nS
        s = state_vec[sidx]
        nk = Kernel.node_kind(game, s)
        (nk == Kernel.TERMINAL || nk == Kernel.CHANCE) && continue

        for a in Kernel.legal_actions(game, s, 1)
            trans_ptr[sa_idx] = length(next_state) + 1
            for entry in Exact.transition_kernel(game, s, a)
                ns = entry[1]
                push!(next_state, Encodings.encode(enc, ns))
                push!(prob, Float64(entry[2]))
                push!(reward, _scalar_reward(entry[3]))
            end
            sa_idx += 1
        end
    end
    trans_ptr[end] = length(next_state) + 1

    return CompiledMDP{typeof(enc),S}(
        nS,
        n_actions_vec,
        action_ptr,
        trans_ptr,
        next_state,
        prob,
        reward,
        enc,
        state_vec,
    )
end

function compile_zero_sum_markov_game(game::Kernel.AbstractGame, states)
    Kernel.num_players(game) == 2 ||
        throw(ArgumentError("compile_zero_sum_markov_game requires a 2-player game."))

    state_vec = collect(states)
    S = eltype(state_vec)
    enc = _build_state_encoder(state_vec)
    nS = length(state_vec)

    first_simultaneous = nothing
    for s in state_vec
        if Kernel.node_kind(game, s) == Kernel.SIMULTANEOUS
            first_simultaneous = s
            break
        end
    end
    first_simultaneous === nothing &&
        throw(ArgumentError("compile_zero_sum_markov_game requires at least one simultaneous state."))

    A1T = typeof(_materialize_actions(Kernel.legal_actions(game, first_simultaneous, 1)))
    A2T = typeof(_materialize_actions(Kernel.legal_actions(game, first_simultaneous, 2)))

    a1s = zeros(Int, nS)
    a2s = zeros(Int, nS)
    pair_ptr = Vector{Int}(undef, nS + 1)
    legal_actions_p1 = Vector{A1T}(undef, nS)
    legal_actions_p2 = Vector{A2T}(undef, nS)

    total_pairs = 0
    @inbounds for sidx in 1:nS
        s = state_vec[sidx]
        nk = Kernel.node_kind(game, s)
        if nk == Kernel.SIMULTANEOUS
            A1 = _materialize_actions(Kernel.legal_actions(game, s, 1))
            A2 = _materialize_actions(Kernel.legal_actions(game, s, 2))
            legal_actions_p1[sidx] = A1
            legal_actions_p2[sidx] = A2
            a1 = _action_count(A1)
            a2 = _action_count(A2)
            a1s[sidx] = a1
            a2s[sidx] = a2
            pair_ptr[sidx] = total_pairs + 1
            total_pairs += a1 * a2
        else
            legal_actions_p1[sidx] = _materialize_actions(Kernel.legal_actions(game, s, 1))
            legal_actions_p2[sidx] = _materialize_actions(Kernel.legal_actions(game, s, 2))
            pair_ptr[sidx] = total_pairs + 1
        end
    end
    pair_ptr[end] = total_pairs + 1

    trans_ptr = Vector{Int}(undef, total_pairs + 1)
    next_state = Int[]
    prob = Float64[]
    reward = Float64[]
    sizehint!(next_state, max(total_pairs, 16))
    sizehint!(prob, max(total_pairs, 16))
    sizehint!(reward, max(total_pairs, 16))

    pair_idx = 1
    @inbounds for sidx in 1:nS
        s = state_vec[sidx]
        Kernel.node_kind(game, s) == Kernel.SIMULTANEOUS || continue

        A1 = legal_actions_p1[sidx]
        A2 = legal_actions_p2[sidx]

        for i in eachindex(A1), j in eachindex(A2)
            trans_ptr[pair_idx] = length(next_state) + 1
            ja = Kernel.JointAction((A1[i], A2[j]))
            for entry in Exact.transition_kernel(game, s, ja)
                ns = entry[1]
                push!(next_state, Encodings.encode(enc, ns))
                push!(prob, Float64(entry[2]))
                push!(reward, _scalar_reward(entry[3]))
            end
            pair_idx += 1
        end
    end
    trans_ptr[end] = length(next_state) + 1

    return CompiledZeroSumMarkovGame{typeof(enc),S,A1T,A2T}(
        nS,
        a1s,
        a2s,
        pair_ptr,
        trans_ptr,
        next_state,
        prob,
        reward,
        enc,
        state_vec,
        legal_actions_p1,
        legal_actions_p2,
    )
end

end