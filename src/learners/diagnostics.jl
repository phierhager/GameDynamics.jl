module LearningDiagnostics

using Statistics
using ..LearningFeedback

export RunningStat
export RunningMean
export LearnerTrace
export push!
export reset!
export mean_value
export cumulative_regret
export average_utility
export average_reward
export empirical_action_histogram!
export empirical_action_frequencies!
export utility_gap

# ----------------------------------------------------------------------
# Running scalar statistics
# ----------------------------------------------------------------------

mutable struct RunningStat{T}
    n::Int
    sum::T
end

RunningStat{T}() where {T} = RunningStat{T}(0, zero(T))
RunningStat() = RunningStat{Float64}()

const RunningMean = RunningStat

function reset!(s::RunningStat{T}) where {T}
    s.n = 0
    s.sum = zero(T)
    return s
end

function push!(s::RunningStat{T}, x) where {T}
    s.n += 1
    s.sum += x
    return s
end

mean_value(s::RunningStat) = s.n == 0 ? 0.0 : s.sum / s.n

# ----------------------------------------------------------------------
# Learner trace
# ----------------------------------------------------------------------

"""
Lightweight trace container for repeated-play diagnostics.

Fields:
- `t`: round count
- `cumulative_utility`
- `best_fixed_utility`: cumulative utility of best fixed action benchmark
- `cumulative_reward`: alias-compatible scalar accumulator
"""
mutable struct LearnerTrace{T}
    t::Int
    cumulative_utility::T
    best_fixed_utility::T
    cumulative_reward::T
end

LearnerTrace{T}() where {T} = LearnerTrace{T}(0, zero(T), zero(T), zero(T))
LearnerTrace() = LearnerTrace{Float64}()

function reset!(tr::LearnerTrace{T}) where {T}
    tr.t = 0
    tr.cumulative_utility = zero(T)
    tr.best_fixed_utility = zero(T)
    tr.cumulative_reward = zero(T)
    return tr
end

"""
Update trace from feedback only.

This path tracks realized utility / reward but not regret unless a benchmark
utility vector is available.
"""
function push!(tr::LearnerTrace{T}, fb::LearningFeedback.AbstractFeedback) where {T}
    u = LearningFeedback.realized_utility(fb)
    tr.t += 1
    tr.cumulative_utility += u
    tr.cumulative_reward += u
    return tr
end

"""
Update trace from full-information feedback, accumulating best-fixed baseline.
"""
function push!(tr::LearnerTrace{T}, fb::LearningFeedback.FullInformationFeedback) where {T}
    u = LearningFeedback.realized_utility(fb)
    uv = LearningFeedback.utility_vector(fb)
    tr.t += 1
    tr.cumulative_utility += u
    tr.cumulative_reward += u
    tr.best_fixed_utility += maximum(uv)
    return tr
end

cumulative_regret(tr::LearnerTrace) = tr.best_fixed_utility - tr.cumulative_utility
average_utility(tr::LearnerTrace) = tr.t == 0 ? 0.0 : tr.cumulative_utility / tr.t
average_reward(tr::LearnerTrace) = tr.t == 0 ? 0.0 : tr.cumulative_reward / tr.t

"""
One-step utility gap for diagnostics.
"""
utility_gap(realized_u, benchmark_u) = benchmark_u - realized_u

# ----------------------------------------------------------------------
# Empirical action-use diagnostics
# ----------------------------------------------------------------------

"""
Increment histogram counts in place from an integer action id.
"""
function empirical_action_histogram!(counts::AbstractVector{<:Integer}, action::Int)
    1 <= action <= length(counts) || throw(BoundsError(counts, action))
    counts[action] += 1
    return counts
end

"""
Normalize integer histogram counts into frequencies in place.
"""
function empirical_action_frequencies!(dest::AbstractVector{Float64},
                                       counts::AbstractVector{<:Integer})
    length(dest) == length(counts) || throw(ArgumentError("Destination length mismatch."))
    total = sum(counts)
    if total > 0
        invt = 1 / total
        @inbounds for i in eachindex(counts)
            dest[i] = counts[i] * invt
        end
    else
        v = 1 / length(dest)
        @inbounds for i in eachindex(dest)
            dest[i] = v
        end
    end
    return dest
end

end