module LearningContexts

using ..LearningInterfaces

export NullContext
export RoundContext
export ObservationContext
export StateContext
export HistoryContext

struct NullContext <: LearningInterfaces.AbstractLearningContext end

struct RoundContext{T} <: LearningInterfaces.AbstractLearningContext
    t::Int
    payload::T
end

struct ObservationContext{O} <: LearningInterfaces.AbstractLearningContext
    observation::O
end

struct StateContext{S} <: LearningInterfaces.AbstractLearningContext
    state::S
end

struct HistoryContext{H} <: LearningInterfaces.AbstractLearningContext
    history::H
end

end