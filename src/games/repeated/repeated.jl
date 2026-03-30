module RepeatedGames

using Random

using ..StrategyInterface
using ..StrategyProfiles
using ..Kernel
using ..NormalForm

export RepeatedNormalFormGame
export RepeatedRoundRecord
export RepeatedHistory
export empty_history
export num_rounds
export current_round
export round_actions
export round_payoffs
export discounted_return
export undiscounted_return
export play_repeated_profile
export play_general_repeated_profile

struct RepeatedNormalFormGame{G<:NormalForm.NormalFormGame}
    stage_game::G
    horizon::Int
    discount::Float64
end

function RepeatedNormalFormGame(stage_game::NormalForm.NormalFormGame;
                                horizon::Int,
                                discount::Real = 1.0)
    horizon > 0 || throw(ArgumentError("horizon must be positive."))
    0.0 < discount <= 1.0 || throw(ArgumentError("discount must be in (0,1]."))
    return RepeatedNormalFormGame{typeof(stage_game)}(stage_game, horizon, Float64(discount))
end

struct RepeatedRoundRecord{N}
    actions::NTuple{N,Int}
    payoffs::NTuple{N,Float64}
end

struct RepeatedHistory{N,R<:AbstractVector{RepeatedRoundRecord{N}}}
    rounds::R
end

RepeatedHistory{N}() where {N} =
    RepeatedHistory{N,Vector{RepeatedRoundRecord{N}}}(RepeatedRoundRecord{N}[])

empty_history(::NormalForm.NormalFormGame{N}) where {N} = RepeatedHistory{N}()
empty_history(g::RepeatedNormalFormGame) = empty_history(g.stage_game)

num_rounds(h::RepeatedHistory) = length(h.rounds)
current_round(h::RepeatedHistory) = length(h.rounds) + 1
round_actions(h::RepeatedHistory, t::Int) = h.rounds[t].actions
round_payoffs(h::RepeatedHistory, t::Int) = h.rounds[t].payoffs

@inline function _push_round!(h::RepeatedHistory{N},
                              actions::NTuple{N,Int},
                              payoffs::NTuple{N,Float64}) where {N}
    push!(h.rounds, RepeatedRoundRecord{N}(actions, payoffs))
    return h
end

function discounted_return(stage_payoffs::AbstractVector, γ::Real)
    acc = 0.0
    coeff = 1.0
    g = Float64(γ)

    for x in stage_payoffs
        acc += coeff * Float64(x)
        coeff *= g
    end

    return acc
end

undiscounted_return(stage_payoffs::AbstractVector) = sum(stage_payoffs)

@inline function _validate_stage_action(stage_game::NormalForm.NormalFormGame, player::Int, action)
    n = stage_game.action_sizes[player]
    action isa Integer && 1 <= action <= n ||
        throw(ArgumentError("Illegal action $action for player $player; expected an integer in 1:$n."))
    return Int(action)
end

@inline function _validate_stage_action_profile(stage_game::NormalForm.NormalFormGame{N},
                                                acts::NTuple{N,Int}) where {N}
    @inbounds for p in 1:N
        _validate_stage_action(stage_game, p, acts[p])
    end
    return acts
end

function play_repeated_profile(g::RepeatedNormalFormGame,
                               profile::StrategyProfiles.StrategyProfile,
                               rng::AbstractRNG = Random.default_rng())
    N = Kernel.num_players(g.stage_game)
    length(profile) == N || throw(ArgumentError("Profile size does not match number of players."))

    totals = ntuple(_ -> 0.0, N)
    coeff = 1.0

    for _ in 1:g.horizon
        ja = StrategyProfiles.sample_joint_action(profile, rng)
        acts = _validate_stage_action_profile(g.stage_game, ja)
        pay = NormalForm.pure_payoff(g.stage_game, acts)
        totals = ntuple(i -> totals[i] + coeff * pay[i], N)
        coeff *= g.discount
    end

    return totals
end

function play_repeated_profile(g::RepeatedNormalFormGame,
                               profile::Tuple,
                               rng::AbstractRNG = Random.default_rng())
    return play_repeated_profile(g, StrategyProfiles.StrategyProfile(profile), rng)
end

function play_general_repeated_profile(g::RepeatedNormalFormGame,
                                       profile::StrategyProfiles.StrategyProfile,
                                       rng::AbstractRNG = Random.default_rng())
    N = Kernel.num_players(g.stage_game)
    length(profile) == N || throw(ArgumentError("Profile size does not match number of players."))

    totals = ntuple(_ -> 0.0, N)
    coeff = 1.0
    history = RepeatedHistory{N}()

    for _ in 1:g.horizon
        acts = ntuple(i -> begin
            a = StrategyInterface.sample_action(profile[i], history, rng)
            _validate_stage_action(g.stage_game, i, a)
        end, N)

        pay = NormalForm.pure_payoff(g.stage_game, acts)
        _push_round!(history, acts, pay)

        totals = ntuple(i -> totals[i] + coeff * pay[i], N)
        coeff *= g.discount
    end

    return totals, history
end

function play_general_repeated_profile(g::RepeatedNormalFormGame,
                                       profile::Tuple,
                                       rng::AbstractRNG = Random.default_rng())
    return play_general_repeated_profile(g, StrategyProfiles.StrategyProfile(profile), rng)
end

end