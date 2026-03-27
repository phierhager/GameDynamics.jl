module Classification

using ..Kernel
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
export is_potential_game
export is_anonymous_game
export is_network_game
export is_stackelberg_game
export is_hierarchical_game
export is_posg
export is_decpomdp
export is_constant_sum

"""
Heuristic lightweight classifiers.

These predicates are convenience labels for dispatch, reporting, and UI.
They are not intended as formal semantic definitions or solver-precondition
checks. Use semantic validators for strong guarantees.
"""
is_normal_form(::Kernel.AbstractGame) = false
is_extensive_form(::Kernel.AbstractGame) = false
is_bayesian_game(::Kernel.AbstractGame) = false

"""
Conservative default: only games explicitly modeled as repeated games
should return true. Do not infer this from horizon/max_steps alone.
"""
is_repeated_game(::Kernel.AbstractGame) = false

is_stochastic_game(game::Kernel.AbstractGame) =
    Spec.game_spec(game).stochastic === true

is_perfect_information(game::Kernel.AbstractGame) =
    Spec.game_spec(game).perfect_information === true

is_imperfect_information(game::Kernel.AbstractGame) =
    Spec.game_spec(game).perfect_information === false

is_zero_sum(game::Kernel.AbstractGame) =
    Spec.game_spec(game).payoff_kind == Spec.ZERO_SUM

is_constant_sum(game::Kernel.AbstractGame) =
    Spec.game_spec(game).payoff_kind == Spec.CONSTANT_SUM

"""
Strict interpretation: only explicitly general-sum games count here.
Constant-sum is distinct.
"""
is_general_sum(game::Kernel.AbstractGame) =
    Spec.game_spec(game).payoff_kind == Spec.GENERAL_SUM

# Conservative defaults
is_complete_information(::Kernel.AbstractGame) = false
is_incomplete_information(::Kernel.AbstractGame) = false

"""
Lightweight organizational classifier.

This is not the same as cooperative reward structure.
"""
is_noncooperative(game::Kernel.AbstractGame) =
    Spec.game_spec(game).cooperative === false

is_signaling_game(::Any) = false
is_anonymous_game(::Any) = false
is_potential_game(::Any) = false
is_network_game(::Any) = false
is_stackelberg_game(::Any) = false
is_hierarchical_game(::Any) = false

"""
Heuristic POSG classifier.

A game is treated as a POSG when it is explicitly:
- stochastic
- imperfect-information
- simultaneous-move
- at least weakly compatible with partial observation metadata
"""
function is_posg(game::Kernel.AbstractGame)
    spec = Spec.game_spec(game)
    return spec.stochastic === true &&
           spec.perfect_information === false &&
           spec.simultaneous_moves === true &&
           (
               spec.observation_kind == Spec.PARTIAL_OBSERVATION ||
               spec.observation_kind == Spec.UNKNOWN_OBSERVATION
           ) &&
           Kernel.num_players(game) >= 2
end

"""
Heuristic Dec-POMDP classifier.

A game is treated as a Dec-POMDP when it satisfies the POSG heuristic and is
explicitly cooperative with shared/identical reward metadata.
"""
function is_decpomdp(game::Kernel.AbstractGame)
    spec = Spec.game_spec(game)
    return is_posg(game) &&
           spec.cooperative === true &&
           (
               spec.reward_sharing == Spec.SHARED_REWARD ||
               spec.reward_sharing == Spec.IDENTICAL_REWARD
           )
end

end