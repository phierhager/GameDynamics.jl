module ExactMarkovSolvers

using JuMP
using HiGHS
import MathOptInterface as MOI

using ..Kernel
using ..Exact
using ..Capabilities
using ..Encodings
using ..CompiledMarkovModels

export reachable_states
export value_iteration_mdp
export greedy_policy_from_values
export shapley_value_iteration_zero_sum

# ----------------------------------------------------------------------
# Generic transition-kernel helpers
# Expected transition-kernel entry format:
#   (next_state, prob, reward)
# where reward is scalar or tuple-like.
# ----------------------------------------------------------------------

@inline _next_state(entry) = entry[1]
@inline _prob(entry) = entry[2]
@inline _reward(entry) = entry[3]
@inline _scalar_reward(r) = r isa Real ? Float64(r) : Float64(r[1])

@inline _state_index(idx::AbstractDict, s) = idx[s]
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

@inline _action_count(xs::Base.OneTo) = length(xs)
@inline _action_count(xs::AbstractVector) = length(xs)
@inline _action_count(xs::Tuple) = length(xs)
@inline _action_count(xs) = length(collect(xs))

@inline _materialize_actions(xs::Base.OneTo) = xs
@inline _materialize_actions(xs::AbstractVector) = xs
@inline _materialize_actions(xs::Tuple) = xs
@inline _materialize_actions(xs) = collect(xs)

# ----------------------------------------------------------------------
# Reachability
# ----------------------------------------------------------------------

function _joint_action_tuples(game::Kernel.AbstractGame, state)
    N = Kernel.num_players(game)
    ranges = ntuple(i -> Kernel.legal_actions(game, state, i), N)
    return Iterators.product(ranges...)
end

"""
Finite-state reachability using exact kernels / exact chance.
This is intended for exact small/medium finite games.
"""
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
            Capabilities.has_chance_outcomes(typeof(game)) === Val(true) ||
                throw(ArgumentError("reachable_states requires exact chance outcomes on chance nodes."))
            for (event, _) in Exact.chance_outcomes(game, state)
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
                Capabilities.has_transition_kernel(typeof(game)) === Val(true) ||
                    throw(ArgumentError("reachable_states requires transition_kernel for decision nodes."))
                for entry in Exact.transition_kernel(game, state, a)
                    ns = _next_state(entry)
                    if !haskey(seen, ns)
                        length(queue) < max_states || error("State budget exceeded in reachable_states.")
                        seen[ns] = length(queue) + 1
                        push!(queue, ns)
                    end
                end
            end

        elseif nk == Kernel.SIMULTANEOUS
            Capabilities.has_transition_kernel(typeof(game)) === Val(true) ||
                throw(ArgumentError("reachable_states requires transition_kernel for simultaneous nodes."))
            for tup in _joint_action_tuples(game, state)
                ja = Kernel.JointAction(tup)
                for entry in Exact.transition_kernel(game, state, ja)
                    ns = _next_state(entry)
                    if !haskey(seen, ns)
                        length(queue) < max_states || error("State budget exceeded in reachable_states.")
                        seen[ns] = length(queue) + 1
                        push!(queue, ns)
                    end
                end
            end
        end
    end

    return Tuple(queue)
end

# ----------------------------------------------------------------------
# Compiled MDP exact value iteration
# ----------------------------------------------------------------------

function value_iteration_mdp(model::CompiledMarkovModels.CompiledMDP;
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

function greedy_policy_from_values(model::CompiledMarkovModels.CompiledMDP,
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

        best_a = 0
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
                best_a = local_a
            end
            local_a += 1
        end

        policy[s] = best_a
    end

    return policy
end

# ----------------------------------------------------------------------
# Generic single-agent value iteration
# ----------------------------------------------------------------------

"""
Exact discounted value iteration for single-agent finite MDPs.

Requirements:
- num_players(game) == 1
- exact transition_kernel returning (next_state, prob, reward)
- states is a finite tuple/list of states

Returns:
    V, state_index
where `state_index` is an encoder-backed indexer usable with
`greedy_policy_from_values`.
"""
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
                for (event, prob) in Exact.chance_outcomes(game, s)
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
                    for entry in Exact.transition_kernel(game, s, a)
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

function _greedy_policy_from_values_indexed(game::Kernel.AbstractGame,
                                            states,
                                            V,
                                            idx;
                                            discount::Float64 = 0.99)
    policy = Vector{Union{Nothing,Int}}(undef, length(states))

    @inbounds for sidx in eachindex(states)
        s = states[sidx]
        nk = Kernel.node_kind(game, s)

        if nk == Kernel.TERMINAL || nk == Kernel.CHANCE
            policy[sidx] = nothing
            continue
        end

        best_a = 0
        best_q = -Inf
        for a in Kernel.legal_actions(game, s, 1)
            q = 0.0
            for entry in Exact.transition_kernel(game, s, a)
                ns = _next_state(entry)
                p = _prob(entry)
                r = _reward(entry)
                nsidx = _state_index(idx, ns)
                rr = _scalar_reward(r)
                q += p * (rr + discount * V[nsidx])
            end
            if q > best_q
                best_q = q
                best_a = a
            end
        end
        policy[sidx] = best_a
    end

    return policy
end

function _greedy_policy_from_values_generic(game::Kernel.AbstractGame,
                                            states,
                                            V,
                                            idx;
                                            discount::Float64 = 0.99)
    policy = Vector{Any}(undef, length(states))

    @inbounds for sidx in eachindex(states)
        s = states[sidx]
        nk = Kernel.node_kind(game, s)

        if nk == Kernel.TERMINAL || nk == Kernel.CHANCE
            policy[sidx] = nothing
            continue
        end

        best_a = nothing
        best_q = -Inf
        for a in Kernel.legal_actions(game, s, 1)
            q = 0.0
            for entry in Exact.transition_kernel(game, s, a)
                ns = _next_state(entry)
                p = _prob(entry)
                r = _reward(entry)
                nsidx = _state_index(idx, ns)
                rr = _scalar_reward(r)
                q += p * (rr + discount * V[nsidx])
            end
            if q > best_q
                best_q = q
                best_a = a
            end
        end
        policy[sidx] = best_a
    end

    return policy
end

function greedy_policy_from_values(game::Kernel.AbstractGame,
                                   states,
                                   V,
                                   idx;
                                   discount::Float64 = 0.99)
    Kernel.num_players(game) == 1 ||
        throw(ArgumentError("greedy_policy_from_values requires a single-player game."))

    if Kernel.action_mode(typeof(game)) === Kernel.IndexedActions
        return _greedy_policy_from_values_indexed(game, states, V, idx; discount = discount)
    end
    return _greedy_policy_from_values_generic(game, states, V, idx; discount = discount)
end

# ----------------------------------------------------------------------
# Compiled 2-player zero-sum Markov games via Shapley iteration
# ----------------------------------------------------------------------

mutable struct _StageLPWorkspace
    stage::Matrix{Float64}
    x_model::Model
    x::Vector{VariableRef}
    v::VariableRef
    col_constraints::Vector{ConstraintRef}
end

function _StageLPWorkspace(max_m::Int, max_n::Int, optimizer)
    stage = zeros(Float64, max_m, max_n)

    model = Model(optimizer)
    set_silent(model)

    @variable(model, x[1:max_m] >= 0.0)
    @variable(model, v)
    @constraint(model, sum(x) == 1.0)
    col_constraints = Vector{ConstraintRef}(undef, max_n)
    for j in 1:max_n
        col_constraints[j] = @constraint(model, sum(stage[i, j] * x[i] for i in 1:max_m) >= v)
    end
    @objective(model, Max, v)

    return _StageLPWorkspace(stage, model, x, v, col_constraints)
end

function _solve_stage_value!(ws::_StageLPWorkspace, m::Int, n::Int)
    model = ws.x_model

    @inbounds for j in 1:length(ws.col_constraints)
        if j <= n
            set_normalized_rhs(ws.col_constraints[j], 0.0)
            set_name(ws.col_constraints[j], "")
        end
    end

    for j in 1:length(ws.col_constraints)
        if j <= n
            con = ws.col_constraints[j]
            f = JuMP.constraint_object(con).func
            terms = f.terms
            empty!(terms)
            @inbounds for i in 1:m
                terms[ws.x[i]] = ws.stage[i, j]
            end
            terms[ws.v] = -1.0
        end
    end

    optimize!(model)
    termination_status(model) == MOI.OPTIMAL ||
        error("Stage LP failed during compiled Shapley iteration.")

    return objective_value(model)
end

function shapley_value_iteration_zero_sum(model::CompiledMarkovModels.CompiledZeroSumMarkovGame;
                                          discount::Float64 = 0.99,
                                          tol::Float64 = 1e-8,
                                          max_iter::Int = 1000,
                                          optimizer = HiGHS.Optimizer)
    nS = model.n_states
    V = zeros(Float64, nS)
    Vnew = similar(V)

    max_m = maximum(model.n_actions_p1)
    max_n = maximum(model.n_actions_p2)
    lpws = _StageLPWorkspace(max(max_m, 1), max(max_n, 1), optimizer)

    for _ in 1:max_iter
        Δ = 0.0

        @inbounds for s in 1:nS
            m = model.n_actions_p1[s]
            n = model.n_actions_p2[s]

            if m == 0 || n == 0
                Vnew[s] = 0.0
            else
                pair_first = model.pair_ptr[s]
                pair_last = model.pair_ptr[s + 1] - 1

                pair_idx = pair_first
                for i in 1:m, j in 1:n
                    t_first = model.trans_ptr[pair_idx]
                    t_last = model.trans_ptr[pair_idx + 1] - 1
                    q = 0.0
                    for t in t_first:t_last
                        ns = model.next_state[t]
                        p = model.prob[t]
                        r = model.reward[t]
                        q += p * (r + discount * V[ns])
                    end
                    lpws.stage[i, j] = q
                    pair_idx += 1
                end

                Vnew[s] = _solve_stage_value!(lpws, m, n)
            end

            Δ = max(Δ, abs(Vnew[s] - V[s]))
        end

        V, Vnew = Vnew, V
        Δ <= tol && break
    end

    return V, model.state_encoder
end

# ----------------------------------------------------------------------
# Generic 2-player zero-sum discounted Markov games via Shapley iteration
# ----------------------------------------------------------------------

"""
Shapley value iteration for finite 2-player zero-sum discounted simultaneous Markov games.

Requirements:
- num_players(game) == 2
- simultaneous or chance nodes only on the reachable state set
- exact transition_kernel returning (next_state, prob, reward)
- player-1 reward defines the zero-sum stage value

For performance, prefer compiling first via:
    model = CompiledMarkovModels.compile_zero_sum_markov_game(game, states)
    shapley_value_iteration_zero_sum(model; ...)
"""
function shapley_value_iteration_zero_sum(game::Kernel.AbstractGame,
                                          states;
                                          discount::Float64 = 0.99,
                                          tol::Float64 = 1e-8,
                                          max_iter::Int = 1000,
                                          optimizer = HiGHS.Optimizer)
    Kernel.num_players(game) == 2 ||
        throw(ArgumentError("shapley_value_iteration_zero_sum requires a 2-player game."))

    if Kernel.action_mode(typeof(game)) === Kernel.IndexedActions
        model = CompiledMarkovModels.compile_zero_sum_markov_game(game, states)
        return shapley_value_iteration_zero_sum(model;
                                                discount = discount,
                                                tol = tol,
                                                max_iter = max_iter,
                                                optimizer = optimizer)
    end

    idx = _build_state_index(states)

    V = zeros(Float64, length(states))
    Vnew = similar(V)

    legal_actions_p1 = Vector{Any}(undef, length(states))
    legal_actions_p2 = Vector{Any}(undef, length(states))
    max_m = 1
    max_n = 1

    @inbounds for sidx in eachindex(states)
        s = states[sidx]
        nk = Kernel.node_kind(game, s)
        if nk == Kernel.SIMULTANEOUS
            A1 = _materialize_actions(Kernel.legal_actions(game, s, 1))
            A2 = _materialize_actions(Kernel.legal_actions(game, s, 2))
            legal_actions_p1[sidx] = A1
            legal_actions_p2[sidx] = A2
            max_m = max(max_m, _action_count(A1))
            max_n = max(max_n, _action_count(A2))
        else
            legal_actions_p1[sidx] = nothing
            legal_actions_p2[sidx] = nothing
        end
    end

    lpws = _StageLPWorkspace(max_m, max_n, optimizer)

    for _ in 1:max_iter
        Δ = 0.0

        @inbounds for sidx in eachindex(states)
            s = states[sidx]
            nk = Kernel.node_kind(game, s)

            if nk == Kernel.TERMINAL
                Vnew[sidx] = 0.0

            elseif nk == Kernel.CHANCE
                acc = 0.0
                for (event, prob) in Exact.chance_outcomes(game, s)
                    ns, r, _ = Kernel.step(game, s, Kernel.ChanceOutcome(event))
                    nsidx = _state_index(idx, ns)
                    rr = _scalar_reward(r)
                    acc += prob * (rr + discount * V[nsidx])
                end
                Vnew[sidx] = acc

            elseif nk == Kernel.SIMULTANEOUS
                A1 = legal_actions_p1[sidx]
                A2 = legal_actions_p2[sidx]
                m = _action_count(A1)
                n = _action_count(A2)

                for i in 1:m, j in 1:n
                    ja = Kernel.JointAction((A1[i], A2[j]))
                    q = 0.0
                    for entry in Exact.transition_kernel(game, s, ja)
                        ns = _next_state(entry)
                        p = _prob(entry)
                        r = _reward(entry)
                        nsidx = _state_index(idx, ns)
                        rr = _scalar_reward(r)
                        q += p * (rr + discount * V[nsidx])
                    end
                    lpws.stage[i, j] = q
                end

                Vnew[sidx] = _solve_stage_value!(lpws, m, n)

            else
                A = Kernel.legal_actions(game, s, Kernel.current_player(game, s))
                p = Kernel.current_player(game, s)

                if p == 1
                    best = -Inf
                    for a in A
                        q = 0.0
                        for entry in Exact.transition_kernel(game, s, a)
                            ns = _next_state(entry)
                            p_ = _prob(entry)
                            r = _reward(entry)
                            nsidx = _state_index(idx, ns)
                            rr = _scalar_reward(r)
                            q += p_ * (rr + discount * V[nsidx])
                        end
                        best = max(best, q)
                    end
                    Vnew[sidx] = best
                else
                    best = Inf
                    for a in A
                        q = 0.0
                        for entry in Exact.transition_kernel(game, s, a)
                            ns = _next_state(entry)
                            p_ = _prob(entry)
                            r = _reward(entry)
                            nsidx = _state_index(idx, ns)
                            rr = _scalar_reward(r)
                            q += p_ * (rr + discount * V[nsidx])
                        end
                        best = min(best, q)
                    end
                    Vnew[sidx] = best
                end
            end

            Δ = max(Δ, abs(Vnew[sidx] - V[sidx]))
        end

        V, Vnew = Vnew, V
        Δ <= tol && break
    end

    return V, idx
end

end