module Runtime

using ..Kernel
using ..Spec

export AbstractEpisodeLimit, NoLimit, StepLimit
export default_episode_limit, is_truncated

abstract type AbstractEpisodeLimit end

struct NoLimit <: AbstractEpisodeLimit end

struct StepLimit <: AbstractEpisodeLimit
    max_steps::Int
    function StepLimit(max_steps::Int)
        max_steps > 0 || throw(ArgumentError("max_steps must be positive, got $max_steps."))
        new(max_steps)
    end
end

"""
Default runtime truncation policy.

This is outside the kernel. By default it consults `Spec.game_spec(game).max_steps`.
"""
function default_episode_limit(game::Kernel.AbstractGame)
    max_steps = Spec.game_spec(game).max_steps
    return isnothing(max_steps) ? NoLimit() : StepLimit(max_steps)
end

is_truncated(::NoLimit, step_count::Int) = false
is_truncated(limit::StepLimit, step_count::Int) = step_count >= limit.max_steps

end