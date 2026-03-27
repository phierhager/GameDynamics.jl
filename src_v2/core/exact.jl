module Exact

using ..Kernel
using ..Spaces

export state_space, observation_space, action_space, indexed_action_space
export information_state, public_observation, public_state
export chance_outcomes, transition_kernel, observation_kernel
export terminal_payoffs

function indexed_action_space(game::Kernel.AbstractGame, player::Int)
    Kernel.action_mode(typeof(game)) === Kernel.IndexedActions ||
        throw(ArgumentError("indexed_action_space is only valid for IndexedActions games."))
    return Spaces.IndexedDiscreteSpace(Kernel.indexed_action_count(game, player))
end

information_state(game::Kernel.AbstractGame, state, player::Int) =
    throw(MethodError(information_state, (game, state, player)))

state_space(game::Kernel.AbstractGame) =
    throw(MethodError(state_space, (game,)))

observation_space(game::Kernel.AbstractGame, player::Int) =
    throw(MethodError(observation_space, (game, player)))

action_space(game::Kernel.AbstractGame, state, player::Int) =
    throw(MethodError(action_space, (game, state, player)))

public_observation(game::Kernel.AbstractGame, state) =
    throw(MethodError(public_observation, (game, state)))

public_state(game::Kernel.AbstractGame, state) =
    throw(MethodError(public_state, (game, state)))

chance_outcomes(game::Kernel.AbstractGame, state) =
    throw(MethodError(chance_outcomes, (game, state)))

transition_kernel(game::Kernel.AbstractGame, state, action) =
    throw(MethodError(transition_kernel, (game, state, action)))

observation_kernel(game::Kernel.AbstractGame, state, action, next_state) =
    throw(MethodError(observation_kernel, (game, state, action, next_state)))

terminal_payoffs(game::Kernel.AbstractGame, state) =
    throw(MethodError(terminal_payoffs, (game, state)))

end