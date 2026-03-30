module HedgeLearners

using Random

using ..LearningInterfaces
using ..RuntimeRecords

export Hedge
export HedgeState

struct Hedge{T} <: LearningInterfaces.AbstractLearner
    eta::T
    n_actions::Int

    function Hedge(eta::T, n_actions::Int) where {T}
        n_actions > 0 || throw(ArgumentError("n_actions must be positive."))
        eta > zero(T) || throw(ArgumentError("eta must be positive."))
        return new{T}(eta, n_actions)
    end
end

mutable struct HedgeState{T} <: LearningInterfaces.AbstractLearnerState
    log_weights::Vector{T}
    probs::Vector{T}

    function HedgeState(l::Hedge{T}) where {T}
        return new{T}(zeros(T, l.n_actions), fill(one(T) / l.n_actions, l.n_actions))
    end
end

const HedgeFullInfoRecord = Union{
    RuntimeRecords.FullInformationRecord,
    RuntimeRecords.ContextFullInformationRecord,
}

LearningInterfaces.action_mode(::Hedge) = :discrete_index
LearningInterfaces.requires_feedback_type(::Hedge) = HedgeFullInfoRecord
LearningInterfaces.supports_action_space(::Hedge) = :finite_discrete

function LearningInterfaces.reset!(l::Hedge, st::HedgeState)
    fill!(st.log_weights, 0)
    fill!(st.probs, 1 / l.n_actions)
    return st
end

function LearningInterfaces.strategy!(dest::AbstractVector,
                                    l::Hedge{T},
                                    st::HedgeState{T},
                                    record::Union{Nothing,RuntimeRecords.AbstractStepRecord} = nothing) where {T}
    length(dest) == l.n_actions || throw(ArgumentError("Destination length mismatch."))
    maxw = maximum(st.log_weights)
    z = zero(T)
    @inbounds for i in 1:l.n_actions
        dest[i] = exp(st.log_weights[i] - maxw)
        z += dest[i]
    end
    invz = one(T) / z
    @inbounds for i in 1:l.n_actions
        dest[i] *= invz
        st.probs[i] = dest[i]
    end
    return dest
end

function LearningInterfaces.act!(l::Hedge,
                                 st::HedgeState,
                                 record::Union{Nothing,RuntimeRecords.AbstractStepRecord} = nothing,
                                 rng::AbstractRNG = Random.default_rng())
    LearningInterfaces.strategy!(st.probs, l, st, record)
    r = rand(rng)
    c = 0.0
    @inbounds for i in eachindex(st.probs)
        c += st.probs[i]
        r <= c && return i
    end
    return last(eachindex(st.probs))
end

function LearningInterfaces.update!(l::Hedge{T},
                                    st::HedgeState{T},
                                    rec::HedgeFullInfoRecord) where {T}
    uv = rec.feedback
    length(uv) == l.n_actions || throw(ArgumentError("Utility vector length mismatch."))
    @inbounds for i in 1:l.n_actions
        st.log_weights[i] += l.eta * uv[i]
    end
    return st
end

end