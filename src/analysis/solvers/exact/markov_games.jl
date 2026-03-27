module ExactMarkovGameSolvers

using JuMP
using HiGHS
import MathOptInterface as MOI

using ..Kernel
using ..TabularMarkovGames
using ..TabularCompile

export shapley_value_iteration_zero_sum

@inline _next_state(entry) = entry[1]
@inline _prob(entry) = entry[2]
@inline _reward(entry) = entry[3]
@inline _scalar_reward(r) = r isa Real ? Float64(r) : Float64(r[1])

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
        error("Stage LP failed during Shapley iteration.")

    return objective_value(lpm.model)
end

function shapley_value_iteration_zero_sum(model::TabularMarkovGames.TabularZeroSumMarkovGame;
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

function shapley_value_iteration_zero_sum(game::Kernel.AbstractGame,
                                          states;
                                          discount::Float64 = 0.99,
                                          tol::Float64 = 1e-8,
                                          max_iter::Int = 1000,
                                          optimizer = HiGHS.Optimizer)
    Kernel.num_players(game) == 2 ||
        throw(ArgumentError("shapley_value_iteration_zero_sum requires a 2-player game."))

    state_vec = collect(states)
    model = TabularCompile.compile_zero_sum_markov_game(game, state_vec)

    return shapley_value_iteration_zero_sum(model;
                                            discount = discount,
                                            tol = tol,
                                            max_iter = max_iter,
                                            optimizer = optimizer)
end

end