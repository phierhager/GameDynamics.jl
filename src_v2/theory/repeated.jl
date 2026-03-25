module RepeatedGames

using Random
using ..Kernel
using ..Strategies
using ..NormalForm
using ..Classification

export RepeatedNormalFormGame
export discounted_return
export undiscounted_return
export play_repeated_profile

struct RepeatedNormalFormGame{G<:NormalForm.NormalFormGame}
    stage_game::G
    horizon::Int
    discount::Float64
end

function RepeatedNormalFormGame(stage_game::NormalForm.NormalFormGame; horizon::Int, discount::Real = 1.0)
    horizon > 0 || throw(ArgumentError("horizon must be positive."))
    0.0 < discount <= 1.0 || throw(ArgumentError("discount must be in (0,1]."))
    return RepeatedNormalFormGame{typeof(stage_game)}(stage_game, horizon, Float64(discount))
end

Classification.is_repeated_game(::RepeatedNormalFormGame) = true

function discounted_return(stage_payoffs::AbstractVector, γ::Real)
    acc = zero(eltype(stage_payoffs))
    coeff = 1.0
    for x in stage_payoffs
        acc += coeff * x
        coeff *= γ
    end
    return acc
end

undiscounted_return(stage_payoffs::AbstractVector) = sum(stage_payoffs)

function play_repeated_profile(g::RepeatedNormalFormGame,
                               profile::Tuple{Vararg{Strategies.AbstractStrategy,N}},
                               rng::AbstractRNG = Random.default_rng()) where {N}
    N == Kernel.num_players(g.stage_game) ||
        throw(ArgumentError("Profile size does not match number of players."))

    totals = ntuple(_ -> 0.0, N)
    coeff = 1.0
    for _ in 1:g.horizon
        ja = Strategies.sample_joint_action(profile, rng)
        pay = NormalForm.pure_payoff(g.stage_game, ja.actions)
        totals = ntuple(i -> totals[i] + coeff * pay[i], N)
        coeff *= g.discount
    end
    return totals
end

play_repeated_profile(g::RepeatedNormalFormGame,
                      profile::Strategies.StrategyProfile{N},
                      rng::AbstractRNG = Random.default_rng()) where {N} =
    play_repeated_profile(g, profile.strategies, rng)

end