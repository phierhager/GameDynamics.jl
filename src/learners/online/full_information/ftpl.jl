module FTPLLearners

using Random
using ..Learning
using ..LearningInterfaces
using ..LearningFeedback

export FTPL
export FTPLState

struct FTPL{T} <: LearningInterfaces.AbstractLearner
    eta::T
    n_actions::Int
    
    function FTPL(eta::T, n_actions::Int) where {T}
        n_actions > 0 || throw(ArgumentError("n_actions must be positive."))
        eta > zero(T) || throw(ArgumentError("eta must be positive."))
        return new{T}(eta, n_actions)
    end
end

mutable struct FTPLState{T} <: LearningInterfaces.AbstractLearnerState
    cumulative_utilities::Vector{T}
    scratch::Vector{T}

    function FTPLState(l::FTPL{T}) where {T}
        return new{T}(zeros(T, l.n_actions), zeros(T, l.n_actions))
    end
end

Learning.learner_family(::FTPL) = :full_information
LearningInterfaces.action_mode(::FTPL) = :discrete_index
LearningInterfaces.requires_feedback_type(::FTPL) = LearningFeedback.FullInformationFeedback
LearningInterfaces.supports_action_space(::FTPL) = :finite_discrete

function LearningInterfaces.reset!(l::FTPL, st::FTPLState)
    fill!(st.cumulative_utilities, 0)
    fill!(st.scratch, 0)
    return st
end

@inline _gumbel(rng::AbstractRNG) = -log(-log(rand(rng)))

function LearningInterfaces.policy!(dest::AbstractVector,
                                    l::FTPL,
                                    st::FTPLState,
                                    ctx::LearningInterfaces.AbstractLearningContext)
    throw(ArgumentError("FTPL has no stable closed-form policy; use `act!` directly."))
end

function LearningInterfaces.act!(l::FTPL{T},
                                 st::FTPLState{T},
                                 ctx::LearningInterfaces.AbstractLearningContext,
                                 rng::AbstractRNG = Random.default_rng()) where {T}
    best_i = 1
    best_v = typemin(T)
    @inbounds for i in 1:l.n_actions
        v = st.cumulative_utilities[i] + l.eta * _gumbel(rng)
        st.scratch[i] = v
        if i == 1 || v > best_v
            best_v = v
            best_i = i
        end
    end
    return best_i
end

function LearningInterfaces.update!(l::FTPL{T},
                                    st::FTPLState{T},
                                    fb::LearningFeedback.FullInformationFeedback) where {T}
    uv = LearningFeedback.utility_vector(fb)
    length(uv) == l.n_actions || throw(ArgumentError("Utility vector length mismatch."))
    @inbounds for i in 1:l.n_actions
        st.cumulative_utilities[i] += uv[i]
    end
    return st
end

end