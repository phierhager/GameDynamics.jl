module Enumerative

using ..Kernel

export chance_outcomes
export transition_kernel
export terminal_payoffs

function chance_outcomes(game::Kernel.AbstractGame, state)
    throw(MethodError(chance_outcomes, (game, state)))
end

function transition_kernel(game::Kernel.AbstractGame, state, action)
    throw(MethodError(transition_kernel, (game, state, action)))
end

function terminal_payoffs(game::Kernel.AbstractGame, state)
    throw(MethodError(terminal_payoffs, (game, state)))
end

end