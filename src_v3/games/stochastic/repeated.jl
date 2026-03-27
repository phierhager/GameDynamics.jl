module RepeatedGames

using Random
using ..Kernel
using ..Strategies
using ..NormalForm
using ..Classification

export RepeatedNormalFormGame
export GeneralRepeatedNormalFormGame

export RepeatedRoundRecord
export RepeatedHistory
export empty_history
export num_rounds, current_round
export round_actions, round_payoffs

export discounted_return
export undiscounted_return

export play_repeated_profile
export play_general_repeated_profile

# ----------------------------------------------------------------------
# Public repeated-history objects
# ----------------------------------------------------------------------

"""
One realized stage of repeated play.
"""
struct RepeatedRoundRecord{N}
    actions::NTuple{N,Int}
    payoffs::NTuple{N,Float64}
end

"""
Public history of realized repeated-play stages.

This is the conditioning object used by the general repeated-game helper.
It records the realized joint action and realized stage payoffs from each
completed round in order.

The intended semantics are public-history repeated play with perfect monitoring
of realized stage actions/payoffs. If you later introduce richer monitoring
models, those should likely use a different history or observation object.
"""
struct RepeatedHistory{N,R<:AbstractVector{RepeatedRoundRecord{N}}}
    rounds::R
end

RepeatedHistory{N}() where {N} =
    RepeatedHistory{N,Vector{RepeatedRoundRecord{N}}}(RepeatedRoundRecord{N}[])

empty_history(::NormalForm.NormalFormGame{N}) where {N} = RepeatedHistory{N}()
empty_history(g::RepeatedNormalFormGame) = empty_history(g.stage_game)
empty_history(g::GeneralRepeatedNormalFormGame) = empty_history(g.stage_game)

num_rounds(h::RepeatedHistory) = length(h.rounds)
current_round(h::RepeatedHistory) = length(h.rounds) + 1

round_actions(h::RepeatedHistory, t::Int) = h.rounds[t].actions
round_payoffs(h::RepeatedHistory, t::Int) = h.rounds[t].payoffs

@inline function _push_round!(h::RepeatedHistory{N}, actions::NTuple{N,Int}, payoffs::NTuple{N,Float64}) where {N}
    push!(h.rounds, RepeatedRoundRecord{N}(actions, payoffs))
    return h
end

# ----------------------------------------------------------------------
# Stationary repeated normal-form game
# ----------------------------------------------------------------------

"""
Repeated play of a stage normal-form game with a fixed per-period strategy profile.

This is not a general repeated-game strategy model:
- strategies are sampled independently each round
- strategies do not condition on play history
- the same stage profile is reused each period
"""
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

# ----------------------------------------------------------------------
# General history-dependent repeated normal-form game
# ----------------------------------------------------------------------

"""
General repeated normal-form game with public-history, nonstationary play.

This object represents the same repeated stage-game skeleton as
`RepeatedNormalFormGame`, but intended for strategy profiles whose local play
may depend on the full realized repeated-game history.

Expected strategy interface in `play_general_repeated_profile`:
- each player's strategy is queried as `Strategies.sample_action(strategy, history, rng)`

where `history` is a `RepeatedHistory`.

This is a theory-level helper for nonstationary, history-dependent repeated play.
"""
struct GeneralRepeatedNormalFormGame{G<:NormalForm.NormalFormGame}
    stage_game::G
    horizon::Int
    discount::Float64
end

function GeneralRepeatedNormalFormGame(stage_game::NormalForm.NormalFormGame;
                                       horizon::Int,
                                       discount::Real = 1.0)
    horizon > 0 || throw(ArgumentError("horizon must be positive."))
    0.0 < discount <= 1.0 || throw(ArgumentError("discount must be in (0,1]."))
    return GeneralRepeatedNormalFormGame{typeof(stage_game)}(stage_game, horizon, Float64(discount))
end

Classification.is_repeated_game(::RepeatedNormalFormGame) = true
Classification.is_repeated_game(::GeneralRepeatedNormalFormGame) = true

# ----------------------------------------------------------------------
# Return helpers
# ----------------------------------------------------------------------

"""
Compute a discounted scalar return from a sequence of stage payoffs.
"""
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

"""
Compute an undiscounted scalar return from a sequence of stage payoffs.
"""
undiscounted_return(stage_payoffs::AbstractVector) = sum(stage_payoffs)

# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

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

# ----------------------------------------------------------------------
# Stationary repeated-play simulator
# ----------------------------------------------------------------------

"""
Simulate repeated play of a fixed stage-game strategy profile.

Returns the tuple of discounted cumulative player returns over the repeated
horizon.
"""
function play_repeated_profile(g::RepeatedNormalFormGame,
                               profile::Tuple{Vararg{Strategies.AbstractStrategy,N}},
                               rng::AbstractRNG = Random.default_rng()) where {N}
    N == Kernel.num_players(g.stage_game) ||
        throw(ArgumentError("Profile size does not match number of players."))

    totals = ntuple(_ -> 0.0, N)
    coeff = 1.0

    for _ in 1:g.horizon
        ja = Strategies.sample_joint_action(profile, rng)
        acts = _validate_stage_action_profile(g.stage_game, Tuple(ja))
        pay = NormalForm.pure_payoff(g.stage_game, acts)
        totals = ntuple(i -> totals[i] + coeff * pay[i], N)
        coeff *= g.discount
    end

    return totals
end

play_repeated_profile(g::RepeatedNormalFormGame,
                      profile::Strategies.StrategyProfile{N},
                      rng::AbstractRNG = Random.default_rng()) where {N} =
    play_repeated_profile(g, profile.strategies, rng)

# ----------------------------------------------------------------------
# General history-dependent repeated-play simulator
# ----------------------------------------------------------------------

"""
Simulate public-history, nonstationary repeated play.

Each local strategy is queried as:
- `Strategies.sample_action(strategy, history, rng)`

where `history` is the realized `RepeatedHistory` accumulated so far.

Returns:
- discounted cumulative payoff tuple
- realized repeated history

as `(totals, history)`.
"""
function play_general_repeated_profile(g::GeneralRepeatedNormalFormGame,
                                       profile::Tuple{Vararg{Strategies.AbstractStrategy,N}},
                                       rng::AbstractRNG = Random.default_rng()) where {N}
    N == Kernel.num_players(g.stage_game) ||
        throw(ArgumentError("Profile size does not match number of players."))

    totals = ntuple(_ -> 0.0, N)
    coeff = 1.0
    history = RepeatedHistory{N}()

    for _ in 1:g.horizon
        acts = ntuple(i -> begin
            a = Strategies.sample_action(profile[i], history, rng)
            _validate_stage_action(g.stage_game, i, a)
        end, N)

        pay = NormalForm.pure_payoff(g.stage_game, acts)
        _push_round!(history, acts, pay)

        totals = ntuple(i -> totals[i] + coeff * pay[i], N)
        coeff *= g.discount
    end

    return totals, history
end

play_general_repeated_profile(g::GeneralRepeatedNormalFormGame,
                              profile::Strategies.StrategyProfile{N},
                              rng::AbstractRNG = Random.default_rng()) where {N} =
    play_general_repeated_profile(g, profile.strategies, rng)

end