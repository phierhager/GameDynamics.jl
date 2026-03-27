module TestMockGames

using Random
using GameLab.Kernel
using GameLab.Spec

export DecisionGame, SimultaneousGame, BadSimultaneousGame, EmptyInitGame
export DummyState

struct DummyState <: Kernel.AbstractState
    tag::Symbol
end

struct DecisionGame <: Kernel.AbstractGame{2,Tuple{Float64,Float64}} end
struct SimultaneousGame <: Kernel.AbstractGame{3,NTuple{3,Float64}} end
struct BadSimultaneousGame <: Kernel.AbstractGame{3,NTuple{3,Float64}} end
struct EmptyInitGame <: Kernel.AbstractGame{1,Float64} end

Kernel.init_state(::DecisionGame, rng::AbstractRNG = Random.default_rng()) = DummyState(:decision)
Kernel.init_state(::SimultaneousGame, rng::AbstractRNG = Random.default_rng()) = DummyState(:simultaneous)
Kernel.init_state(::BadSimultaneousGame, rng::AbstractRNG = Random.default_rng()) = DummyState(:bad_simultaneous)

Kernel.node_kind(::DecisionGame, ::DummyState) = Kernel.DECISION
Kernel.node_kind(::SimultaneousGame, ::DummyState) = Kernel.SIMULTANEOUS
Kernel.node_kind(::BadSimultaneousGame, ::DummyState) = Kernel.SIMULTANEOUS

Kernel.current_player(::DecisionGame, ::DummyState) = 2

Kernel.active_players(::SimultaneousGame, ::DummyState) = (1, 3)
Kernel.active_players(::BadSimultaneousGame, ::DummyState) = (3, 2)

Kernel.legal_actions(::SimultaneousGame, ::DummyState, p::Int) =
    p == 1 ? (10, 11) : p == 3 ? (20, 21) : ()

Kernel.legal_actions(::BadSimultaneousGame, ::DummyState, p::Int) =
    p == 2 ? (5, 6) : p == 3 ? (7, 8) : ()

Spec.game_spec(::DecisionGame) = Spec.GameSpec(
    perfect_information = true,
    stochastic = false,
    simultaneous_moves = false,
    cooperative = false,
    observation_kind = Spec.FULL_STATE_OBSERVATION,
    payoff_kind = Spec.ZERO_SUM,
)

Spec.game_spec(::SimultaneousGame) = Spec.GameSpec(
    perfect_information = false,
    stochastic = true,
    simultaneous_moves = true,
    cooperative = true,
    observation_kind = Spec.PARTIAL_OBSERVATION,
    reward_sharing = Spec.SHARED_REWARD,
)

Spec.game_spec(::BadSimultaneousGame) = Spec.GameSpec(
    perfect_information = false,
    stochastic = true,
    simultaneous_moves = true,
    cooperative = true,
    observation_kind = Spec.PARTIAL_OBSERVATION,
    reward_sharing = Spec.SHARED_REWARD,
)

end