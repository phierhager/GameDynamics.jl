module LearningFeedback

export AbstractFeedback
export AbstractObservation
export NoObservation

export FullInformationFeedback
export BanditFeedback
export SemiBanditFeedback
export GradientFeedback
export TrajectoryFeedback

export chosen_action
export realized_utility
export observation
export utility_vector
export gradient
export trajectory
export return_value
export is_full_information
export is_bandit_feedback
export is_gradient_feedback

abstract type AbstractFeedback end
abstract type AbstractObservation end

struct NoObservation <: AbstractObservation end

struct FullInformationFeedback{A,U,V,O<:AbstractObservation} <: AbstractFeedback
    chosen_action::A
    realized_utility::U
    utility_vector::V
    observation::O
end

struct BanditFeedback{A,U,O<:AbstractObservation} <: AbstractFeedback
    chosen_action::A
    realized_utility::U
    observation::O
end

struct SemiBanditFeedback{A,U,C,O<:AbstractObservation} <: AbstractFeedback
    chosen_action::A
    realized_utility::U
    component_feedback::C
    observation::O
end

struct GradientFeedback{A,U,G,O<:AbstractObservation} <: AbstractFeedback
    chosen_action::A
    realized_utility::U
    gradient::G
    observation::O
end

struct TrajectoryFeedback{T,R,O<:AbstractObservation} <: AbstractFeedback
    trajectory::T
    return_value::R
    observation::O
end

chosen_action(f::FullInformationFeedback) = f.chosen_action
chosen_action(f::BanditFeedback) = f.chosen_action
chosen_action(f::SemiBanditFeedback) = f.chosen_action
chosen_action(f::GradientFeedback) = f.chosen_action

realized_utility(f::FullInformationFeedback) = f.realized_utility
realized_utility(f::BanditFeedback) = f.realized_utility
realized_utility(f::SemiBanditFeedback) = f.realized_utility
realized_utility(f::GradientFeedback) = f.realized_utility

observation(f::FullInformationFeedback) = f.observation
observation(f::BanditFeedback) = f.observation
observation(f::SemiBanditFeedback) = f.observation
observation(f::GradientFeedback) = f.observation
observation(f::TrajectoryFeedback) = f.observation

utility_vector(f::FullInformationFeedback) = f.utility_vector
gradient(f::GradientFeedback) = f.gradient

trajectory(f::TrajectoryFeedback) = f.trajectory
return_value(f::TrajectoryFeedback) = f.return_value

is_full_information(::AbstractFeedback) = false
is_full_information(::FullInformationFeedback) = true

is_bandit_feedback(::AbstractFeedback) = false
is_bandit_feedback(::BanditFeedback) = true
is_bandit_feedback(::SemiBanditFeedback) = true

is_gradient_feedback(::AbstractFeedback) = false
is_gradient_feedback(::GradientFeedback) = true

FullInformationFeedback(chosen_action, realized_utility, utility_vector) =
    FullInformationFeedback(chosen_action, realized_utility, utility_vector, NoObservation())

BanditFeedback(chosen_action, realized_utility) =
    BanditFeedback(chosen_action, realized_utility, NoObservation())

GradientFeedback(chosen_action, realized_utility, grad) =
    GradientFeedback(chosen_action, realized_utility, grad, NoObservation())

end