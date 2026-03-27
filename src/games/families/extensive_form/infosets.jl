module ExtensiveFormInfosets

using ...Kernel
using ..ExtensiveFormInformationStates

export infoset
export infoset_kind
export uses_information_state
export require_information_state_interface

"""
Return whether the game declares a dedicated information-state interface.
"""
@inline uses_information_state(game::Kernel.AbstractGame) =
    ExtensiveFormInformationStates.uses_information_state(game)

"""
Require that the game exposes a dedicated information-state interface.
"""
function require_information_state_interface(game::Kernel.AbstractGame)
    ExtensiveFormInformationStates.require_information_state_interface(game)
    return game
end

"""
Describe how `infoset(game, state, player)` will be formed.

Possible values:
- `:information_state`
- `:observation_fallback`
"""
function infoset_kind(game::Kernel.AbstractGame)
    return uses_information_state(game) ? :information_state : :observation_fallback
end

"""
Canonical infoset accessor.

Semantics:
- if the game declares a dedicated information-state interface, use
  `ExtensiveFormInformationStates.information_state(game, state, player)`
- otherwise fall back to `Kernel.observe(game, state, player)`

This is a semantic convenience helper. Solver-grade imperfect-information
algorithms may wish to require the dedicated information-state interface
instead of accepting the observation fallback.
"""
function infoset(game::Kernel.AbstractGame, state, player::Int)
    if uses_information_state(game)
        return ExtensiveFormInformationStates.information_state(game, state, player)
    end
    return Kernel.observe(game, state, player)
end

end