module EXP3Learners

using Random
using ..LearningInterfaces
using ..RuntimeRecords

export EXP3
export EXP3State

struct EXP3{T} <: LearningInterfaces.AbstractLearner
    gamma::T
    eta::T
    n_actions::Int

    function EXP3(gamma::T, eta::T, n_actions::Int) where {T}
        n_actions > 0 || throw(ArgumentError("n_actions must be positive."))
        zero(T) <= gamma <= one(T) || throw(ArgumentError("gamma must be in [0,1]."))
        eta > zero(T) || throw(ArgumentError("eta must be positive."))
        return new{T}(gamma, eta, n_actions)
    end
end

mutable struct EXP3State{T} <: LearningInterfaces.AbstractLearnerState
    log_weights::Vector{T}
    probs::Vector{T}

    function EXP3State(l::EXP3{T}) where {T}
        return new{T}(zeros(T, l.n_actions), fill(one(T) / l.n_actions, l.n_actions))
    end
end

const EXP3BanditRecord = Union{
    RuntimeRecords.BanditRecord,
    RuntimeRecords.ContextBanditRecord,
}

LearningInterfaces.action_mode(::EXP3) = :discrete_index
LearningInterfaces.requires_feedback_type(::EXP3) = EXP3BanditRecord
LearningInterfaces.supports_action_space(::EXP3) = :finite_discrete

function LearningInterfaces.reset!(l::EXP3, st::EXP3State)
    fill!(st.log_weights, 0)
    fill!(st.probs, 1 / l.n_actions)
    return st
end

function LearningInterfaces.strategy!(dest::AbstractVector,
                                    l::EXP3{T},
                                    st::EXP3State{T},
                                    record::Union{Nothing,RuntimeRecords.AbstractStepRecord} = nothing) where {T}
    length(dest) == l.n_actions || throw(ArgumentError("Destination length mismatch."))
    maxw = maximum(st.log_weights)
    z = zero(T)
    @inbounds for i in 1:l.n_actions
        dest[i] = exp(st.log_weights[i] - maxw)
        z += dest[i]
    end
    invz = one(T) / z
    unif = one(T) / l.n_actions
    @inbounds for i in 1:l.n_actions
        dest[i] = (one(T) - l.gamma) * (dest[i] * invz) + l.gamma * unif
        st.probs[i] = dest[i]
    end
    return dest
end

function LearningInterfaces.act!(l::EXP3,
                                 st::EXP3State,
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

function LearningInterfaces.update!(l::EXP3{T},
                                    st::EXP3State{T},
                                    rec::EXP3BanditRecord) where {T}
    a = rec.action
    u = rec.reward
    p = st.probs[a]
    st.log_weights[a] += l.eta * (u / max(p, eps(T)))
    return st
end

end