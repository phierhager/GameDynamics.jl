module FTRLLearners

using Random
using ..Learning
using ..LearningInterfaces
using ..LearningFeedback

export EntropicFTRL
export EntropicFTRLState

struct EntropicFTRL{T} <: LearningInterfaces.AbstractLearner
    eta::T
    n_actions::Int

    function EntropicFTRL(eta::T, n_actions::Int) where {T}
        n_actions > 0 || throw(ArgumentError("n_actions must be positive."))
        eta > zero(T) || throw(ArgumentError("eta must be positive."))
        return new{T}(eta, n_actions)
    end
end

mutable struct EntropicFTRLState{T} <: LearningInterfaces.AbstractLearnerState
    cumulative_utilities::Vector{T}
    probs::Vector{T}

    function EntropicFTRLState(l::EntropicFTRL{T}) where {T}
        return new{T}(zeros(T, l.n_actions), fill(one(T) / l.n_actions, l.n_actions))
    end
end

Learning.learner_family(::EntropicFTRL) = :full_information
LearningInterfaces.action_mode(::EntropicFTRL) = :discrete_index
LearningInterfaces.requires_feedback_type(::EntropicFTRL) = LearningFeedback.FullInformationFeedback
LearningInterfaces.supports_action_space(::EntropicFTRL) = :finite_discrete

function LearningInterfaces.reset!(l::EntropicFTRL, st::EntropicFTRLState)
    fill!(st.cumulative_utilities, 0)
    fill!(st.probs, 1 / l.n_actions)
    return st
end

function LearningInterfaces.policy!(dest::AbstractVector,
                                    l::EntropicFTRL{T},
                                    st::EntropicFTRLState{T},
                                    ctx::LearningInterfaces.AbstractLearningContext) where {T}
    length(dest) == l.n_actions || throw(ArgumentError("Destination length mismatch."))
    maxu = maximum(st.cumulative_utilities)
    z = zero(T)
    @inbounds for i in 1:l.n_actions
        dest[i] = exp(l.eta * (st.cumulative_utilities[i] - maxu))
        z += dest[i]
    end
    invz = one(T) / z
    @inbounds for i in 1:l.n_actions
        dest[i] *= invz
        st.probs[i] = dest[i]
    end
    return dest
end

function LearningInterfaces.act!(l::EntropicFTRL,
                                 st::EntropicFTRLState,
                                 ctx::LearningInterfaces.AbstractLearningContext,
                                 rng::AbstractRNG = Random.default_rng())
    LearningInterfaces.policy!(st.probs, l, st, ctx)
    r = rand(rng)
    c = 0.0
    @inbounds for i in eachindex(st.probs)
        c += st.probs[i]
        if r <= c
            return i
        end
    end
    return last(eachindex(st.probs))
end

function LearningInterfaces.update!(l::EntropicFTRL{T},
                                    st::EntropicFTRLState{T},
                                    fb::LearningFeedback.FullInformationFeedback) where {T}
    uv = LearningFeedback.utility_vector(fb)
    length(uv) == l.n_actions || throw(ArgumentError("Utility vector length mismatch."))
    @inbounds for i in 1:l.n_actions
        st.cumulative_utilities[i] += uv[i]
    end
    return st
end

end