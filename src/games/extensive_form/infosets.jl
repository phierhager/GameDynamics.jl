module ExtensiveFormInfosets

using ..Kernel
using ..ExtensiveFormInformationStates

export infoset
export infoset_kind
export uses_information_state
export require_information_state_interface

@inline uses_information_state(game::Kernel.AbstractGame) =
    ExtensiveFormInformationStates.uses_information_state(game)

function require_information_state_interface(game::Kernel.AbstractGame)
    ExtensiveFormInformationStates.require_information_state_interface(game)
    return game
end

infoset_kind(game::Kernel.AbstractGame) =
    uses_information_state(game) ? :information_state : :observation

function infoset(game::Kernel.AbstractGame, state, player::Int)
    if uses_information_state(game)
        return ExtensiveFormInformationStates.information_state(game, state, player)
    end
    return Kernel.observe(game, state, player)
end

end