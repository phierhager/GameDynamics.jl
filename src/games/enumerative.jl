module Enumerative

using ..Kernel

export chance_outcomes
export transition_kernel
export terminal_payoffs

"""
Enumerate chance outcomes available at a chance node.

Expected return:
- iterable of `(event, probability)`
"""
function chance_outcomes(game::Kernel.AbstractGame, state)
    throw(MethodError(chance_outcomes, (game, state)))
end

"""
Enumerate the transition kernel induced by a player-controlled action.

Expected return:
- iterable of `(next_state, probability, reward)`

This is for exact/tabular methods.
It is separate from `Kernel.step`, which only returns one realized transition.
"""
function transition_kernel(game::Kernel.AbstractGame, state, action)
    throw(MethodError(transition_kernel, (game, state, action)))
end

"""
Return terminal payoffs for a terminal state.

Expected return:
- scalar reward for single-player problems
- tuple/vector reward for multi-player problems
"""
function terminal_payoffs(game::Kernel.AbstractGame, state)
    throw(MethodError(terminal_payoffs, (game, state)))
end

end