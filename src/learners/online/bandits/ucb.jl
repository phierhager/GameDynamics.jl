module UCBLearners

using Random
using ..Learning
using ..LearningInterfaces
using ..LearningFeedback

export UCB1
export UCB1State

struct UCB1{T} <: LearningInterfaces.AbstractLearner
    c::T
    n_actions::Int

    function UCB1(c::T, n_actions::Int) where {T}
        n_actions > 0 || throw(ArgumentError("n_actions must be positive."))
        c >= zero(T) || throw(ArgumentError("c must be nonnegative."))
        return UCB1{T}(c, n_actions)
    end
end

mutable struct UCB1State{T} <: LearningInterfaces.AbstractLearnerState
    counts::Vector{Int}
    value_sums::Vector{T}
    round::Int

    function UCB1State(l::UCB1{T}) where {T}
        return UCB1State(zeros(Int, l.n_actions), zeros(T, l.n_actions), 0)
    end
end

Learning.learner_family(::UCB1) = :bandit
LearningInterfaces.action_mode(::UCB1) = :discrete_index
LearningInterfaces.requires_feedback_type(::UCB1) = LearningFeedback.BanditFeedback
LearningInterfaces.supports_action_space(::UCB1) = :finite_discrete

function LearningInterfaces.reset!(l::UCB1, st::UCB1State)
    fill!(st.counts, 0)
    fill!(st.value_sums, 0)
    st.round = 0
    return st
end

function LearningInterfaces.policy!(dest::AbstractVector,
                                    l::UCB1,
                                    st::UCB1State,
                                    ctx::LearningInterfaces.AbstractLearningContext)
    throw(ArgumentError("UCB1 is an index policy; use `act!` directly."))
end

function LearningInterfaces.act!(l::UCB1{T},
                                 st::UCB1State{T},
                                 ctx::LearningInterfaces.AbstractLearningContext,
                                 rng::AbstractRNG = Random.default_rng()) where {T}
    st.round += 1

    @inbounds for a in 1:l.n_actions
        if st.counts[a] == 0
            return a
        end
    end

    best_a = 1
    best_v = typemin(T)
    logt = log(st.round)

    @inbounds for a in 1:l.n_actions
        μ = st.value_sums[a] / st.counts[a]
        bonus = l.c * sqrt(logt / st.counts[a])
        v = μ + bonus
        if a == 1 || v > best_v
            best_v = v
            best_a = a
        end
    end

    return best_a
end

function LearningInterfaces.update!(l::UCB1{T},
                                    st::UCB1State{T},
                                    fb::LearningFeedback.BanditFeedback) where {T}
    a = LearningFeedback.chosen_action(fb)
    u = LearningFeedback.realized_utility(fb)
    st.counts[a] += 1
    st.value_sums[a] += u
    return st
end

end