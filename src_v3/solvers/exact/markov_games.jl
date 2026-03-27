module ExactMarkovGameSolvers

using JuMP
using HiGHS
import MathOptInterface as MOI

using ..Kernel
using ..Exact
using ..Encodings
using ..CompiledMarkovModels

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
# Compiled 2-player zero-sum Markov games via Shapley iteration
# ----------------------------------------------------------------------

mutable struct _StageLPModel
    model::Model
    x::Vector{VariableRef}
    v::VariableRef
    col_constraints::Vector{ConstraintRef}
end

mutable struct _StageLPWorkspace{O}
    stage::Matrix{Float64}
    cache::Dict{Tuple{Int,Int},_StageLPModel}
    optimizer_factory::O
end

function _StageLPWorkspace(max_m::Int, max_n::Int, optimizer)
    return _StageLPWorkspace(
        zeros(Float64, max_m, max_n),
        Dict{Tuple{Int,Int},_StageLPModel}(),
        optimizer,
    )
end

function _get_stage_lp!(ws::_StageLPWorkspace, m::Int, n::Int)
    key = (m, n)
    return get!(ws.cache, key) do
        model = Model(ws.optimizer_factory)
        set_silent(model)

        @variable(model, x[1:m] >= 0.0)
        @variable(model, v)
        @constraint(model, sum(x) == 1.0)

        col_constraints = Vector{ConstraintRef}(undef, n)
        for j in 1:n
            col_constraints[j] = @constraint(model, sum(ws.stage[i, j] * x[i] for i in 1:m) >= v)
        end

        @objective(model, Max, v)
        _StageLPModel(model, x, v, col_constraints)
    end
end

function _solve_stage_value!(ws::_StageLPWorkspace, m::Int, n::Int)
    lpm = _get_stage_lp!(ws, m, n)

    @inbounds for j in 1:n
        con = lpm.col_constraints[j]
        for i in 1:m
            set_normalized_coefficient(con, lpm.x[i], ws.stage[i, j])
        end
        set_normalized_coefficient(con, lpm.v, -1.0)
    end

    optimize!(lpm.model)
    termination_status(lpm.model) == MOI.OPTIMAL ||
        error("Stage LP failed during compiled Shapley iteration.")

    return objective_value(lpm.model)
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

@inline function _can_use_compiled_zero_sum_path(game::Kernel.AbstractGame, states)
    Kernel.action_mode(typeof(game)) === Kernel.IndexedActions || return false
    @inbounds for s in states
        nk = Kernel.node_kind(game, s)
        if !(nk == Kernel.SIMULTANEOUS || nk == Kernel.TERMINAL)
            return false
        end
    end
    return true
end

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

    state_vec = collect(states)

    if _can_use_compiled_zero_sum_path(game, state_vec)
        model = CompiledMarkovModels.compile_zero_sum_markov_game(game, state_vec)
        return shapley_value_iteration_zero_sum(model;
                                                discount = discount,
                                                tol = tol,
                                                max_iter = max_iter,
                                                optimizer = optimizer)
    end

    idx = _build_state_index(state_vec)

    V = zeros(Float64, length(state_vec))
    Vnew = similar(V)

    legal_actions_p1 = Vector{Any}(undef, length(state_vec))
    legal_actions_p2 = Vector{Any}(undef, length(state_vec))
    max_m = 1
    max_n = 1

    @inbounds for sidx in eachindex(state_vec)
        s = state_vec[sidx]
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

        @inbounds for sidx in eachindex(state_vec)
            s = state_vec[sidx]
            nk = Kernel.node_kind(game, s)

            if nk == Kernel.TERMINAL
                Vnew[sidx] = 0.0

            elseif nk == Kernel.CHANCE
                acc = 0.0
                for (event, prob_) in Exact.chance_outcomes(game, s)
                    ns, r, _ = Kernel.step(game, s, Kernel.ChanceOutcome(event))
                    nsidx = _state_index(idx, ns)
                    rr = _scalar_reward(r)
                    acc += prob_ * (rr + discount * V[nsidx])
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