module Capabilities

using ..Kernel

export has_action_mask
export has_chance_outcomes
export has_transition_kernel
export has_observation_kernel
export has_information_state
export has_public_observation
export has_public_state
export has_counterfactual_values
export has_transition_observation
export has_state_space
export has_action_space
export has_observation_space
export supports_population_model

has_action_mask(::Type{<:Kernel.AbstractGame}) = Val(false)
has_chance_outcomes(::Type{<:Kernel.AbstractGame}) = Val(false)
has_transition_kernel(::Type{<:Kernel.AbstractGame}) = Val(false)
has_observation_kernel(::Type{<:Kernel.AbstractGame}) = Val(false)
has_information_state(::Type{<:Kernel.AbstractGame}) = Val(false)
has_public_observation(::Type{<:Kernel.AbstractGame}) = Val(false)
has_public_state(::Type{<:Kernel.AbstractGame}) = Val(false)
has_counterfactual_values(::Type{<:Kernel.AbstractGame}) = Val(false)
has_transition_observation(::Type{<:Kernel.AbstractGame}) = Val(false)

has_state_space(::Type{<:Kernel.AbstractGame}) = Val(false)
has_action_space(::Type{<:Kernel.AbstractGame}) = Val(false)
has_observation_space(::Type{<:Kernel.AbstractGame}) = Val(false)

supports_population_model(::Type{<:Kernel.AbstractGame}) = Val(false)

end