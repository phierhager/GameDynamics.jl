module Classification

using ..Kernel
using ..Capabilities
using ..Spec

export is_normal_form
export is_extensive_form
export is_bayesian_game
export is_repeated_game
export is_stochastic_game
export is_perfect_information
export is_imperfect_information
export is_complete_information
export is_incomplete_information
export is_zero_sum
export is_general_sum
export is_noncooperative
export is_signaling_game
export is_congestion_game
export is_potential_game
export is_anonymous_game
export is_graphical_game
export is_network_game
export is_stackelberg_game
export is_hierarchical_game
export is_posg
export is_decpomdp

"""
Heuristic lightweight classifiers.

These predicates are convenience predicates, not formal semantics.
Concrete theory/game modules should extend them for specific types where appropriate.
"""

is_normal_form(::Kernel.AbstractGame) = false
is_extensive_form(::Kernel.AbstractGame) = false
is_bayesian_game(::Kernel.AbstractGame) = false
is_repeated_game(game::Kernel.AbstractGame) =
    Spec.game_spec(game).horizon_kind == Spec.CONTINUING ||
    (!isnothing(Spec.game_spec(game).max_steps) && Spec.game_spec(game).max_steps > 1)

is_stochastic_game(game::Kernel.AbstractGame) = Spec.game_spec(game).stochastic

is_perfect_information(game::Kernel.AbstractGame) = Spec.game_spec(game).perfect_information
is_imperfect_information(game::Kernel.AbstractGame) = !is_perfect_information(game)

# Conservative defaults
is_complete_information(::Kernel.AbstractGame) = false
is_incomplete_information(::Kernel.AbstractGame) = false

is_zero_sum(game::Kernel.AbstractGame) = Spec.game_spec(game).zero_sum
is_general_sum(game::Kernel.AbstractGame) = Spec.game_spec(game).general_sum
is_noncooperative(::Kernel.AbstractGame) = true

is_signaling_game(::Any) = false
is_congestion_game(::Any) = false
is_potential_game(::Any) = false
is_anonymous_game(::Any) = false
is_graphical_game(::Any) = false
is_network_game(::Any) = false
is_stackelberg_game(::Any) = false
is_hierarchical_game(::Any) = false

function is_posg(game::Kernel.AbstractGame)
    return Spec.game_spec(game).stochastic &&
           (Capabilities.has_information_state(typeof(game)) === Val(true))
end

is_decpomdp(::Any) = false

end