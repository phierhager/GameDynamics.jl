module Exact

using ..Kernel
using ..Spaces

export state_space, observation_space, action_space, indexed_action_space
export information_state, public_observation, public_state
export chance_outcomes, transition_kernel, observation_kernel
export terminal_payoffs
export has_public_observation

state_space(game::Kernel.AbstractGame) =
    error("state_space not implemented for $(typeof(game)).")

observation_space(game::Kernel.AbstractGame, player::Int) =
    error("observation_space not implemented for $(typeof(game)).")

action_space(game::Kernel.AbstractGame, state, player::Int) =
    error("action_space not implemented for $(typeof(game)).")

function indexed_action_space(game::Kernel.AbstractGame, player::Int)
    Kernel.action_mode(typeof(game)) === Kernel.IndexedActions ||
        throw(ArgumentError("indexed_action_space is only valid for IndexedActions games."))
    return Spaces.IndexedDiscreteSpace(Kernel.indexed_action_count(game, player))
end

information_state(game::Kernel.AbstractGame, state, player::Int) =
    error("information_state not implemented for $(typeof(game)).")

has_public_observation(::Type{<:Kernel.AbstractGame}) = false

public_observation(game::Kernel.AbstractGame, state) =
    error("public_observation not implemented for $(typeof(game)).")

public_state(game::Kernel.AbstractGame, state) =
    error("public_state not implemented for $(typeof(game)).")

chance_outcomes(game::Kernel.AbstractGame, state) =
    error("chance_outcomes not implemented for $(typeof(game)).")

transition_kernel(game::Kernel.AbstractGame, state, action) =
    error("transition_kernel not implemented for $(typeof(game)).")

observation_kernel(game::Kernel.AbstractGame, state, action, next_state) =
    error("observation_kernel not implemented for $(typeof(game)).")

terminal_payoffs(game::Kernel.AbstractGame, state) =
    error("terminal_payoffs not implemented for $(typeof(game)).")

end