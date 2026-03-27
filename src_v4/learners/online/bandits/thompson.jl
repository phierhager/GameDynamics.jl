module ThompsonLearners

using Random
using ..Learning
using ..LearningInterfaces
using ..LearningFeedback

export GaussianThompson
export GaussianThompsonState

struct GaussianThompson{T} <: LearningInterfaces.AbstractLearner
    prior_mean::T
    prior_precision::T
    obs_precision::T
    n_actions::Int
end

mutable struct GaussianThompsonState{T} <: LearningInterfaces.AbstractLearnerState
    posterior_mean::Vector{T}
    posterior_precision::Vector{T}
end

function GaussianThompson(prior_mean::T,
                          prior_precision::T,
                          obs_precision::T,
                          n_actions::Int) where {T}
    n_actions > 0 || throw(ArgumentError("n_actions must be positive."))
    prior_precision > zero(T) || throw(ArgumentError("prior_precision must be positive."))
    obs_precision > zero(T) || throw(ArgumentError("obs_precision must be positive."))
    return GaussianThompson{T}(prior_mean, prior_precision, obs_precision, n_actions)
end

function GaussianThompsonState(l::GaussianThompson{T}) where {T}
    return GaussianThompsonState(fill(l.prior_mean, l.n_actions),
                                 fill(l.prior_precision, l.n_actions))
end

Learning.learner_family(::GaussianThompson) = :bayesian_bandit
LearningInterfaces.action_mode(::GaussianThompson) = :discrete_index
LearningInterfaces.requires_feedback_type(::GaussianThompson) = LearningFeedback.BanditFeedback
LearningInterfaces.supports_action_space(::GaussianThompson) = :finite_discrete

function LearningInterfaces.reset!(l::GaussianThompson, st::GaussianThompsonState)
    fill!(st.posterior_mean, l.prior_mean)
    fill!(st.posterior_precision, l.prior_precision)
    return st
end

function LearningInterfaces.policy!(dest::AbstractVector,
                                    l::GaussianThompson,
                                    st::GaussianThompsonState,
                                    ctx::LearningInterfaces.AbstractLearningContext)
    throw(ArgumentError("GaussianThompson is a posterior-sampling policy; use `act!` directly."))
end

function LearningInterfaces.act!(l::GaussianThompson{T},
                                 st::GaussianThompsonState{T},
                                 ctx::LearningInterfaces.AbstractLearningContext,
                                 rng::AbstractRNG = Random.default_rng()) where {T}
    best_a = 1
    best_v = typemin(T)

    @inbounds for a in 1:l.n_actions
        σ = inv(sqrt(st.posterior_precision[a]))
        sample = st.posterior_mean[a] + σ * randn(rng)
        if a == 1 || sample > best_v
            best_v = sample
            best_a = a
        end
    end

    return best_a
end

function _gaussian_posterior_update!(μ::T, τ::T, y::T, obsτ::T) where {T}
    τnew = τ + obsτ
    μnew = (τ * μ + obsτ * y) / τnew
    return μnew, τnew
end

function LearningInterfaces.update!(l::GaussianThompson{T},
                                    st::GaussianThompsonState{T},
                                    fb::LearningFeedback.BanditFeedback) where {T}
    a = LearningFeedback.chosen_action(fb)
    y = LearningFeedback.realized_utility(fb)
    μ, τ = _gaussian_posterior_update!(st.posterior_mean[a], st.posterior_precision[a], y, l.obs_precision)
    st.posterior_mean[a] = μ
    st.posterior_precision[a] = τ
    return st
end

end