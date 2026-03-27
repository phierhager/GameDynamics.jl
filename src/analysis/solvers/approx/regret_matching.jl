module RegretMatchingSolvers

using ..TabularMatrixGames
using ..ApproxSolverCommon

export RegretMatchingWorkspace
export reset!
export regret_matching!
export regret_matching
export current_strategy!
export current_strategy
export average_strategy
export average_strategy!

mutable struct RegretMatchingWorkspace
    regret1::Vector{Float64}
    regret2::Vector{Float64}
    strat_sum1::Vector{Float64}
    strat_sum2::Vector{Float64}
    strategy1::Vector{Float64}
    strategy2::Vector{Float64}
    util1::Vector{Float64}
    util2::Vector{Float64}
end

function RegretMatchingWorkspace(game::TabularMatrixGames.TabularMatrixGame)
    m = game.n_actions_p1
    n = game.n_actions_p2
    return RegretMatchingWorkspace(
        zeros(m),
        zeros(n),
        zeros(m),
        zeros(n),
        fill(1 / m, m),
        fill(1 / n, n),
        zeros(m),
        zeros(n),
    )
end

function reset!(ws::RegretMatchingWorkspace)
    m = length(ws.strategy1)
    n = length(ws.strategy2)

    fill!(ws.regret1, 0.0)
    fill!(ws.regret2, 0.0)
    fill!(ws.strat_sum1, 0.0)
    fill!(ws.strat_sum2, 0.0)
    fill!(ws.strategy1, 1 / m)
    fill!(ws.strategy2, 1 / n)
    fill!(ws.util1, 0.0)
    fill!(ws.util2, 0.0)

    return ws
end

ApproxSolverCommon.reset_solver!(ws::RegretMatchingWorkspace) = reset!(ws)

function _regret_match!(σ::Vector{Float64}, r::Vector{Float64})
    s = 0.0
    @inbounds for i in eachindex(r)
        σ[i] = max(r[i], 0.0)
        s += σ[i]
    end
    if s > 0
        invs = 1 / s
        @inbounds for i in eachindex(σ)
            σ[i] *= invs
        end
    else
        v = 1 / length(σ)
        @inbounds for i in eachindex(σ)
            σ[i] = v
        end
    end
    return σ
end

function current_strategy!(dest::Vector{Float64}, regrets::Vector{Float64})
    length(dest) == length(regrets) || throw(ArgumentError("Destination length mismatch."))
    s = 0.0
    @inbounds for i in eachindex(regrets)
        dest[i] = max(regrets[i], 0.0)
        s += dest[i]
    end
    if s > 0
        invs = 1 / s
        @inbounds for i in eachindex(dest)
            dest[i] *= invs
        end
    else
        v = 1 / length(dest)
        @inbounds for i in eachindex(dest)
            dest[i] = v
        end
    end
    return dest
end

function current_strategy(regrets::Vector{Float64})
    dest = similar(regrets)
    return current_strategy!(dest, regrets)
end

function regret_matching!(game::TabularMatrixGames.TabularMatrixGame,
                          workspace::RegretMatchingWorkspace;
                          n_iter::Int = 10000)
    ApproxSolverCommon.require_tabular_2p_matrix_game(game)

    U1 = game.payoff_p1
    U2 = game.payoff_p2

    σ1 = workspace.strategy1
    σ2 = workspace.strategy2
    r1 = workspace.regret1
    r2 = workspace.regret2
    s1 = workspace.strat_sum1
    s2 = workspace.strat_sum2
    u1 = workspace.util1
    u2 = workspace.util2

    for _ in 1:n_iter
        @inbounds for i in eachindex(s1)
            s1[i] += σ1[i]
        end
        @inbounds for j in eachindex(s2)
            s2[j] += σ2[j]
        end

        @inbounds for i in axes(U1, 1)
            acc = 0.0
            for j in axes(U1, 2)
                acc += U1[i, j] * σ2[j]
            end
            u1[i] = acc
        end
        v1 = 0.0
        @inbounds for i in eachindex(σ1)
            v1 += σ1[i] * u1[i]
        end
        @inbounds for i in eachindex(r1)
            r1[i] += u1[i] - v1
        end

        @inbounds for j in axes(U2, 2)
            acc = 0.0
            for i in axes(U2, 1)
                acc += U2[i, j] * σ1[i]
            end
            u2[j] = acc
        end
        v2 = 0.0
        @inbounds for j in eachindex(σ2)
            v2 += σ2[j] * u2[j]
        end
        @inbounds for j in eachindex(r2)
            r2[j] += u2[j] - v2
        end

        _regret_match!(σ1, r1)
        _regret_match!(σ2, r2)
    end

    return workspace
end

function regret_matching(game::TabularMatrixGames.TabularMatrixGame;
                         n_iter::Int = 10000,
                         workspace::RegretMatchingWorkspace = RegretMatchingWorkspace(game))
    reset!(workspace)
    return regret_matching!(game, workspace; n_iter = n_iter)
end

ApproxSolverCommon.run_solver!(game::TabularMatrixGames.TabularMatrixGame,
                               ws::RegretMatchingWorkspace;
                               n_iter::Int = 1_000) =
    regret_matching!(game, ws; n_iter = n_iter)

function average_strategy!(dest::Vector{Float64}, sumσ::Vector{Float64})
    length(dest) == length(sumσ) || throw(ArgumentError("Destination length mismatch."))
    z = sum(sumσ)
    if z > 0
        invz = 1 / z
        @inbounds for i in eachindex(sumσ)
            dest[i] = sumσ[i] * invz
        end
    else
        v = 1 / length(sumσ)
        @inbounds for i in eachindex(sumσ)
            dest[i] = v
        end
    end
    return dest
end

function average_strategy(sumσ::Vector{Float64})
    dest = similar(sumσ)
    return average_strategy!(dest, sumσ)
end

function ApproxSolverCommon.average_policy!(dest::Vector{Float64},
                                            ws::RegretMatchingWorkspace,
                                            player::Int)
    if player == 1
        return average_strategy!(dest, ws.strat_sum1)
    elseif player == 2
        return average_strategy!(dest, ws.strat_sum2)
    else
        throw(ArgumentError("RegretMatchingWorkspace only supports players 1 and 2."))
    end
end

function ApproxSolverCommon.current_policy!(dest::Vector{Float64},
                                            ws::RegretMatchingWorkspace,
                                            player::Int)
    if player == 1
        length(dest) == length(ws.regret1) || throw(ArgumentError("Destination length mismatch."))
        return current_strategy!(dest, ws.regret1)
    elseif player == 2
        length(dest) == length(ws.regret2) || throw(ArgumentError("Destination length mismatch."))
        return current_strategy!(dest, ws.regret2)
    else
        throw(ArgumentError("RegretMatchingWorkspace only supports players 1 and 2."))
    end
end

end