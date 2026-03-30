module RuntimeRollouts

using Random
using ..Kernel
using ..RuntimeEnvironment
using ..RuntimeStrategyExecution

export rollout_value

@inline _reward_component(r::Tuple, p::Int) = Float64(r[p])
@inline _reward_component(r::AbstractVector, p::Int) = Float64(r[p])

@inline function _reward_component(r::Real, p::Int, N::Int)
    N == 1 || throw(ArgumentError("Scalar reward is only valid for single-player games."))
    return p == 1 ? Float64(r) : 0.0
end

"""
Roll out a strategy profile in a game from the initial state.

Episode truncation follows the runtime episode semantics of the game via
`RuntimeEnvironment.default_episode_limit(game)`. This helper does not accept a
custom rollout-specific step limit; use
`RuntimeEnvironment.GameEnv(...; limit=...)` when you need explicit limit control.
"""
function rollout_value(game::Kernel.AbstractGame,
                       profile;
                       rng::AbstractRNG = Random.default_rng(),
                       discount::Float64 = 1.0)
    0.0 <= discount <= 1.0 || throw(ArgumentError("discount must be in [0,1]."))

    state = Kernel.init_state(game, rng)
    limit = RuntimeEnvironment.default_episode_limit(game)

    N = Kernel.num_players(game)
    totals = zeros(Float64, N)
    coeff = 1.0
    steps = 0

    while !Kernel.is_terminal(game, state) && !RuntimeEnvironment.is_truncated(limit, steps)
        action = RuntimeStrategyExecution.sample_joint_action(profile, game, state, rng)
        state, rewards = Kernel.step(game, state, action, rng)

        @inbounds for i in 1:N
            totals[i] += coeff * (rewards isa Real ?
                _reward_component(rewards, i, N) :
                _reward_component(rewards, i))
        end

        coeff *= discount
        steps += 1
    end

    return ntuple(i -> totals[i], N)
end

end