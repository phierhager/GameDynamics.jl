module ExtensiveFormInformationStates

using ...Kernel

export information_state
export uses_information_state
export require_information_state_interface

function information_state(game::Kernel.AbstractGame, state, player::Int)
    throw(MethodError(information_state, (game, state, player)))
end

uses_information_state(::Type{<:Kernel.AbstractGame}) = false
uses_information_state(game::Kernel.AbstractGame) = uses_information_state(typeof(game))

function require_information_state_interface(game::Kernel.AbstractGame)
    uses_information_state(game) || throw(ArgumentError(
        "Game does not implement a dedicated information-state interface."
    ))
    return game
end

end