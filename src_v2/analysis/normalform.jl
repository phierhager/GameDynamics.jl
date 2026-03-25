module NormalFormAnalysis

using ..Analysis
using ..NormalForm
using ..Strategies

export expected_payoff_profile
export best_response_value
export best_response_gap
export nash_gap
export exploitability_2p_zero_sum
export correlated_deviation_gains
export coarse_correlated_deviation_gains
export is_correlated_equilibrium
export is_coarse_correlated_equilibrium

expected_payoff(game::NormalForm.NormalFormGame, profile) =
    NormalForm.expected_payoff(game, profile)

function best_response_value(game::NormalForm.NormalFormGame{N},
                             player::Int,
                             profile::Tuple{Vararg{Strategies.FiniteMixedStrategy,N}}) where {N}
    _, v = NormalForm.best_response(game, player, profile)
    return v
end

function best_response_gap(game::NormalForm.NormalFormGame{N},
                           player::Int,
                           profile::Tuple{Vararg{Strategies.FiniteMixedStrategy,N}}) where {N}
    u = NormalForm.expected_payoff(game, profile)
    br = best_response_value(game, player, profile)
    return br - u[player]
end

function nash_gap(game::NormalForm.NormalFormGame{N},
                  profile::Tuple{Vararg{Strategies.FiniteMixedStrategy,N}}) where {N}
    u = NormalForm.expected_payoff(game, profile)
    gaps = ntuple(i -> begin
        _, brv = NormalForm.best_response(game, i, profile)
        brv - u[i]
    end, N)
    return maximum(gaps), gaps
end

function exploitability_2p_zero_sum(game::NormalForm.NormalFormGame{2},
                                    profile::Tuple{Strategies.FiniteMixedStrategy,Strategies.FiniteMixedStrategy})
    u = NormalForm.expected_payoff(game, profile)
    br1 = best_response_value(game, 1, profile)
    br2 = best_response_value(game, 2, profile)
    return (br1 - u[1]) + (br2 - u[2])
end

function correlated_deviation_gains(game::NormalForm.NormalFormGame{N},
                                    corr::Strategies.CorrelatedStrategy) where {N}
    support = Strategies.support(corr)
    probs = Strategies.probabilities(corr)

    result = ntuple(p -> zeros(Float64, game.action_sizes[p], game.action_sizes[p]), N)

    @inbounds for k in eachindex(probs)
        profile = support[k]
        mass = probs[k]

        for p in 1:N
            rec = profile[p]
            base_payoff = game.payoffs[p][profile...]
            for dev in 1:game.action_sizes[p]
                dev == rec && continue
                dev_profile = Base.setindex(profile, dev, p)
                gain = Float64(game.payoffs[p][dev_profile...]) - Float64(base_payoff)
                result[p][rec, dev] += mass * gain
            end
        end
    end

    return result
end

function coarse_correlated_deviation_gains(game::NormalForm.NormalFormGame{N},
                                           corr::Strategies.CorrelatedStrategy) where {N}
    support = Strategies.support(corr)
    probs = Strategies.probabilities(corr)

    result = ntuple(p -> zeros(Float64, game.action_sizes[p]), N)

    @inbounds for k in eachindex(probs)
        profile = support[k]
        mass = probs[k]

        for p in 1:N
            base_payoff = game.payoffs[p][profile...]
            for dev in 1:game.action_sizes[p]
                dev_profile = Base.setindex(profile, dev, p)
                gain = Float64(game.payoffs[p][dev_profile...]) - Float64(base_payoff)
                result[p][dev] += mass * gain
            end
        end
    end

    return result
end

function is_correlated_equilibrium(game::NormalForm.NormalFormGame{N},
                                   corr::Strategies.CorrelatedStrategy;
                                   atol::Float64 = 1e-10) where {N}
    gains = correlated_deviation_gains(game, corr)
    for p in 1:N
        G = gains[p]
        @inbounds for i in axes(G, 1), j in axes(G, 2)
            if G[i, j] > atol
                return false
            end
        end
    end
    return true
end

function is_coarse_correlated_equilibrium(game::NormalForm.NormalFormGame{N},
                                          corr::Strategies.CorrelatedStrategy;
                                          atol::Float64 = 1e-10) where {N}
    gains = coarse_correlated_deviation_gains(game, corr)
    for p in 1:N
        g = gains[p]
        @inbounds for i in eachindex(g)
            if g[i] > atol
                return false
            end
        end
    end
    return true
end

Analysis.analysis_family(::Module) = :normal_form

end