module LearningAnalysis

using ..LearningDiagnostics
using ..LearningFeedback

export summarize_trace
export trace_snapshot
export regret_series_value
export average_reward_value
export average_utility_value
export action_frequency_report!

"""
Small named snapshot of a learner trace.
"""
Base.@kwdef struct TraceSnapshot
    rounds::Int
    cumulative_utility::Float64
    cumulative_reward::Float64
    best_fixed_utility::Float64
    cumulative_regret::Float64
    average_utility::Float64
    average_reward::Float64
end

function trace_snapshot(tr::LearningDiagnostics.LearnerTrace)
    return TraceSnapshot(
        rounds = tr.t,
        cumulative_utility = Float64(tr.cumulative_utility),
        cumulative_reward = Float64(tr.cumulative_reward),
        best_fixed_utility = Float64(tr.best_fixed_utility),
        cumulative_regret = Float64(LearningDiagnostics.cumulative_regret(tr)),
        average_utility = Float64(LearningDiagnostics.average_utility(tr)),
        average_reward = Float64(LearningDiagnostics.average_reward(tr)),
    )
end

summarize_trace(tr::LearningDiagnostics.LearnerTrace) = trace_snapshot(tr)

regret_series_value(tr::LearningDiagnostics.LearnerTrace) =
    LearningDiagnostics.cumulative_regret(tr)

average_reward_value(tr::LearningDiagnostics.LearnerTrace) =
    LearningDiagnostics.average_reward(tr)

average_utility_value(tr::LearningDiagnostics.LearnerTrace) =
    LearningDiagnostics.average_utility(tr)

"""
Convert integer action counts into normalized frequencies.

This is a thin analysis-layer wrapper around `LearningDiagnostics.empirical_action_frequencies!`.
"""
function action_frequency_report!(dest::AbstractVector{Float64},
                                  counts::AbstractVector{<:Integer})
    return LearningDiagnostics.empirical_action_frequencies!(dest, counts)
end

end