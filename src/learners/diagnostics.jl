module LearningDiagnostics

using Statistics
using ..RuntimeRecords

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

@inline function _scalar_reward(r)
    r isa Real || throw(ArgumentError("Expected scalar reward, got $(typeof(r))."))
    return Float64(r)
end

@inline function _realized_utility(rec)
    if hasproperty(rec, :reward)
        return _scalar_reward(getproperty(rec, :reward))
    elseif hasproperty(rec, :feedback) && hasproperty(rec, :action)
        fb = getproperty(rec, :feedback)
        a = getproperty(rec, :action)
        return Float64(fb[a])
    else
        throw(ArgumentError("Cannot infer realized utility from record type $(typeof(rec))."))
    end
end

@inline function _best_fixed_utility(rec)
    hasproperty(rec, :feedback) || return nothing
    fb = getproperty(rec, :feedback)
    return Float64(maximum(fb))
end

"""
Update trace from runtime record only.
Tracks realized utility / reward, and tracks best-fixed baseline when a
full-information `feedback` field is available.
"""
function push!(tr::LearnerTrace{T}, rec::RuntimeRecords.AbstractStepRecord) where {T}
    u = _realized_utility(rec)
    tr.t += 1
    tr.cumulative_utility += u
    tr.cumulative_reward += u

    best = _best_fixed_utility(rec)
    if !isnothing(best)
        tr.best_fixed_utility += best
    end

    return tr
end

cumulative_regret(tr::LearnerTrace) = tr.best_fixed_utility - tr.cumulative_utility
average_utility(tr::LearnerTrace) = tr.t == 0 ? 0.0 : tr.cumulative_utility / tr.t
average_reward(tr::LearnerTrace) = tr.t == 0 ? 0.0 : tr.cumulative_reward / tr.t

utility_gap(realized_u, benchmark_u) = benchmark_u - realized_u

function empirical_action_histogram!(counts::AbstractVector{<:Integer}, action::Int)
    1 <= action <= length(counts) || throw(BoundsError(counts, action))
    counts[action] += 1
    return counts
end

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