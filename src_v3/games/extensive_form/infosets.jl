module ExtensiveFormInfosets

using ...Kernel
using ...ModelsExact
using ...ModelsExactValidation

export infoset
export uses_information_state
export require_information_state_interface
export infoset_kind

"""
Return whether the game declares a dedicated information-state interface.
"""
@inline uses_information_state(game::Kernel.AbstractGame) =
    ModelsExactValidation.supports_interface(
        game,
        ModelsExactValidation.InformationStateInterface,
    )

"""
Require that the game exposes a validated information-state interface.
"""
function require_information_state_interface(game::Kernel.AbstractGame)
    ModelsExactValidation.ensure_interface(
        game,
        ModelsExactValidation.InformationStateInterface,
    )
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
- if the game declares `InformationStateInterface`, validate and use
  `ModelsExact.information_state(game, state, player)`
- otherwise fall back to `Kernel.observe(game, state, player)`

This is a semantic convenience helper. Solver-grade imperfect-information
algorithms may wish to require the dedicated information-state interface
instead of accepting the observation fallback.
"""
function infoset(game::Kernel.AbstractGame, state, player::Int)
    if uses_information_state(game)
        require_information_state_interface(game)
        return ModelsExact.information_state(game, state, player)
    end
    return Kernel.observe(game, state, player)
end

end