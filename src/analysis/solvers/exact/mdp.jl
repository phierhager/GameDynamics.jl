module ExactMDPSolvers

using ..Kernel
using ..Exact
using ..Encodings
using ..TabularMDPs

export reachable_states
export value_iteration_mdp
export greedy_policy_from_values

@inline _next_state(entry) = entry[1]
@inline _prob(entry) = entry[2]
@inline _reward(entry) = entry[3]
@inline _scalar_reward(r) = r isa Real ? Float64(r) : Float64(r[1])

@inline _state_index(enc::Encodings.DenseEncoder, s) = Encodings.encode(enc, s)

function _build_state_index(states)
    T = eltype(states)
    enc = Encodings.DenseEncoder{T}()
    Encodings.sizehint!(enc, length(states))
    @inbounds for s in states
        Encodings.encode!(enc, s)
    end
    return enc
end

function reachable_states(game::Kernel.AbstractGame, root_state; max_states::Int = 100000)
    seen = Dict{typeof(root_state),Int}()
    queue = Vector{typeof(root_state)}()
    push!(queue, root_state)
    seen[root_state] = 1

    head = 1
    while head <= length(queue)
        state = queue[head]
        head += 1

        nk = Kernel.node_kind(game, state)
        nk == Kernel.TERMINAL && continue

        if nk == Kernel.CHANCE
            for (event, _) in Enumerative.chance_outcomes(game, state)
                next_state, _, _ = Kernel.step(game, state, Kernel.ChanceOutcome(event))
                if !haskey(seen, next_state)
                    length(queue) < max_states || error("State budget exceeded in reachable_states.")
                    seen[next_state] = length(queue) + 1
                    push!(queue, next_state)
                end
            end

        elseif nk == Kernel.DECISION
            p = Kernel.current_player(game, state)
            for a in Kernel.legal_actions(game, state, p)
                for entry in Enumerative.transition_kernel(game, state, a)
                    ns = _next_state(entry)
                    if !haskey(seen, ns)
                        length(queue) < max_states || error("State budget exceeded in reachable_states.")
                        seen[ns] = length(queue) + 1
                        push!(queue, ns)
                    end
                end
            end

        elseif nk == Kernel.SIMULTANEOUS
            throw(ArgumentError("reachable_states in ExactMDPSolvers is for MDP-style use. Simultaneous nodes belong in markov_games.jl."))
        end
    end

    return Tuple(queue)
end

function value_iteration_mdp(model::TabularMDPs.TabularMDP;
                             discount::Float64 = 0.99,
                             tol::Float64 = 1e-8,
                             max_iter::Int = 10000)
    nS = model.n_states
    V = zeros(Float64, nS)
    Vnew = similar(V)

    for _ in 1:max_iter
        Δ = 0.0

        @inbounds for s in 1:nS
            a_first = model.action_ptr[s]
            a_last = model.action_ptr[s + 1] - 1

            if a_last < a_first
                Vnew[s] = 0.0
            else
                best = -Inf
                for sa in a_first:a_last
                    t_first = model.trans_ptr[sa]
                    t_last = model.trans_ptr[sa + 1] - 1
                    q = 0.0
                    for t in t_first:t_last
                        ns = model.next_state[t]
                        p = model.prob[t]
                        r = model.reward[t]
                        q += p * (r + discount * V[ns])
                    end
                    best = max(best, q)
                end
                Vnew[s] = best
            end

            Δ = max(Δ, abs(Vnew[s] - V[s]))
        end

        V, Vnew = Vnew, V
        Δ <= tol && break
    end

    return V, model.state_encoder
end

function value_iteration_mdp(game::Kernel.AbstractGame,
                             states;
                             discount::Float64 = 0.99,
                             tol::Float64 = 1e-8,
                             max_iter::Int = 10000)
    Kernel.num_players(game) == 1 ||
        throw(ArgumentError("value_iteration_mdp requires a single-player game."))

    idx = _build_state_index(states)
    V = zeros(Float64, length(states))
    Vnew = similar(V)

    for _ in 1:max_iter
        Δ = 0.0

        @inbounds for sidx in eachindex(states)
            s = states[sidx]
            nk = Kernel.node_kind(game, s)

            if nk == Kernel.TERMINAL
                Vnew[sidx] = 0.0

            elseif nk == Kernel.CHANCE
                acc = 0.0
                for (event, prob) in Enumerative.chance_outcomes(game, s)
                    ns, r, _ = Kernel.step(game, s, Kernel.ChanceOutcome(event))
                    nsidx = _state_index(idx, ns)
                    rr = _scalar_reward(r)
                    acc += prob * (rr + discount * V[nsidx])
                end
                Vnew[sidx] = acc

            else
                best = -Inf
                for a in Kernel.legal_actions(game, s, 1)
                    q = 0.0
                    for entry in Enumerative.transition_kernel(game, s, a)
                        ns = _next_state(entry)
                        p = _prob(entry)
                        r = _reward(entry)
                        nsidx = _state_index(idx, ns)
                        rr = _scalar_reward(r)
                        q += p * (rr + discount * V[nsidx])
                    end
                    best = max(best, q)
                end
                Vnew[sidx] = best
            end

            Δ = max(Δ, abs(Vnew[sidx] - V[sidx]))
        end

        V, Vnew = Vnew, V
        Δ <= tol && break
    end

    return V, idx
end

function greedy_policy_from_values(model::TabularMDPs.TabularMDP,
                                   V;
                                   discount::Float64 = 0.99)
    nS = model.n_states
    policy = Vector{Union{Nothing,Int}}(undef, nS)

    @inbounds for s in 1:nS
        a_first = model.action_ptr[s]
        a_last = model.action_ptr[s + 1] - 1

        if a_last < a_first
            policy[s] = nothing
            continue
        end

        best_a = nothing
        best_q = -Inf
        local_a = 1

        for sa in a_first:a_last
            t_first = model.trans_ptr[sa]
            t_last = model.trans_ptr[sa + 1] - 1
            q = 0.0
            for t in t_first:t_last
                ns = model.next_state[t]
                p = model.prob[t]
                r = model.reward[t]
                q += p * (r + discount * V[ns])
            end
            if q > best_q
                best_q = q
                best_a = model.action_label[a_first + local_a - 1]
            end
            local_a += 1
        end

        policy[s] = best_a
    end

    return policy
end

end