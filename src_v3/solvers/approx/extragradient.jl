module FirstOrderMatrixGameSolvers

using ..CompiledNormalFormModels
using ..ApproxSolverCommon

export SimplexProjectionWorkspace
export ExtragradientWorkspace
export reset!
export projected_simplex!
export extragradient_zero_sum!
export extragradient_zero_sum

mutable struct SimplexProjectionWorkspace
    sorted::Vector{Float64}
    cssv::Vector{Float64}
end

SimplexProjectionWorkspace(n::Int) =
    SimplexProjectionWorkspace(zeros(n), zeros(n))

mutable struct ExtragradientWorkspace
    x::Vector{Float64}
    y::Vector{Float64}
    xmid::Vector{Float64}
    ymid::Vector{Float64}
    gx::Vector{Float64}
    gy::Vector{Float64}
    sumx::Vector{Float64}
    sumy::Vector{Float64}
    wx::SimplexProjectionWorkspace
    wy::SimplexProjectionWorkspace
end

function ExtragradientWorkspace(game::CompiledNormalFormModels.CompiledMatrixGame)
    m = game.n_actions_p1
    n = game.n_actions_p2
    return ExtragradientWorkspace(
        fill(1 / m, m),
        fill(1 / n, n),
        zeros(m),
        zeros(n),
        zeros(m),
        zeros(n),
        zeros(m),
        zeros(n),
        SimplexProjectionWorkspace(m),
        SimplexProjectionWorkspace(n),
    )
end

function reset!(ws::ExtragradientWorkspace)
    fill!(ws.x, 1 / length(ws.x))
    fill!(ws.y, 1 / length(ws.y))
    fill!(ws.xmid, 0.0)
    fill!(ws.ymid, 0.0)
    fill!(ws.gx, 0.0)
    fill!(ws.gy, 0.0)
    fill!(ws.sumx, 0.0)
    fill!(ws.sumy, 0.0)
    return ws
end

ApproxSolverCommon.reset_solver!(ws::ExtragradientWorkspace) = reset!(ws)

function projected_simplex!(x::Vector{Float64}, ws::SimplexProjectionWorkspace)
    n = length(x)
    length(ws.sorted) >= n || resize!(ws.sorted, n)
    length(ws.cssv) >= n || resize!(ws.cssv, n)

    @inbounds for i in 1:n
        ws.sorted[i] = x[i]
    end
    sort!(@view(ws.sorted[1:n]); rev = true)

    acc = 0.0
    ρ = 1
    @inbounds for i in 1:n
        acc += ws.sorted[i]
        ws.cssv[i] = acc
        t = (acc - 1.0) / i
        if ws.sorted[i] - t > 0
            ρ = i
        end
    end

    θ = (ws.cssv[ρ] - 1.0) / ρ
    @inbounds for i in 1:n
        x[i] = max(x[i] - θ, 0.0)
    end
    return x
end

function extragradient_zero_sum!(game::CompiledNormalFormModels.CompiledMatrixGame,
                                 ws::ExtragradientWorkspace;
                                 n_iter::Int = 5000,
                                 η::Float64 = 0.1)
    ApproxSolverCommon.require_compiled_2p_matrix_game(game)

    U = game.payoff_p1
    x = ws.x
    y = ws.y
    xmid = ws.xmid
    ymid = ws.ymid
    gx = ws.gx
    gy = ws.gy
    sumx = ws.sumx
    sumy = ws.sumy
    wx = ws.wx
    wy = ws.wy

    m, n = size(U)

    for _ in 1:n_iter
        @inbounds for i in 1:m
            sumx[i] += x[i]
        end
        @inbounds for j in 1:n
            sumy[j] += y[j]
        end

        @inbounds for i in 1:m
            acc = 0.0
            for j in 1:n
                acc += U[i, j] * y[j]
            end
            gx[i] = -acc
        end
        @inbounds for j in 1:n
            acc = 0.0
            for i in 1:m
                acc += U[i, j] * x[i]
            end
            gy[j] = acc
        end

        @inbounds for i in 1:m
            xmid[i] = x[i] - η * gx[i]
        end
        projected_simplex!(xmid, wx)

        @inbounds for j in 1:n
            ymid[j] = y[j] - η * gy[j]
        end
        projected_simplex!(ymid, wy)

        @inbounds for i in 1:m
            acc = 0.0
            for j in 1:n
                acc += U[i, j] * ymid[j]
            end
            gx[i] = -acc
        end
        @inbounds for j in 1:n
            acc = 0.0
            for i in 1:m
                acc += U[i, j] * xmid[i]
            end
            gy[j] = acc
        end

        @inbounds for i in 1:m
            x[i] -= η * gx[i]
        end
        projected_simplex!(x, wx)

        @inbounds for j in 1:n
            y[j] -= η * gy[j]
        end
        projected_simplex!(y, wy)
    end

    return ws
end

function extragradient_zero_sum(game::CompiledNormalFormModels.CompiledMatrixGame;
                                n_iter::Int = 5000,
                                η::Float64 = 0.1,
                                workspace::ExtragradientWorkspace = ExtragradientWorkspace(game))
    reset!(workspace)
    extragradient_zero_sum!(game, workspace; n_iter = n_iter, η = η)
    return workspace.x, workspace.y
end

ApproxSolverCommon.run_solver!(game::CompiledNormalFormModels.CompiledMatrixGame,
                               ws::ExtragradientWorkspace;
                               n_iter::Int = 1_000,
                               η::Float64 = 0.1) =
    extragradient_zero_sum!(game, ws; n_iter = n_iter, η = η)

function ApproxSolverCommon.average_policy!(dest::Vector{Float64},
                                            ws::ExtragradientWorkspace,
                                            player::Int)
    if player == 1
        z = sum(ws.sumx)
        if z > 0
            invz = 1 / z
            @inbounds for i in eachindex(dest)
                dest[i] = ws.sumx[i] * invz
            end
        else
            copyto!(dest, ws.x)
        end
        return dest
    elseif player == 2
        z = sum(ws.sumy)
        if z > 0
            invz = 1 / z
            @inbounds for i in eachindex(dest)
                dest[i] = ws.sumy[i] * invz
            end
        else
            copyto!(dest, ws.y)
        end
        return dest
    else
        throw(ArgumentError("ExtragradientWorkspace only supports players 1 and 2."))
    end
end

function ApproxSolverCommon.current_policy!(dest::Vector{Float64},
                                            ws::ExtragradientWorkspace,
                                            player::Int)
    if player == 1
        length(dest) == length(ws.x) || throw(ArgumentError("Destination length mismatch."))
        copyto!(dest, ws.x)
        return dest
    elseif player == 2
        length(dest) == length(ws.y) || throw(ArgumentError("Destination length mismatch."))
        copyto!(dest, ws.y)
        return dest
    else
        throw(ArgumentError("ExtragradientWorkspace only supports players 1 and 2."))
    end
end

end