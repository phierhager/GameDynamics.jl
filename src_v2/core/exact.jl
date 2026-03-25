module Exact

using ..Kernel
using ..Spaces

export state_space, observation_space, action_space, base_action_space
export information_state, public_observation, public_state
export chance_outcomes, transition_kernel, observation_kernel
export observe_transition
export counterfactual_values
export terminal_payoffs

"""
Optional metadata spaces.
"""
state_space(game::Kernel.AbstractGame) =
    error("state_space not implemented for $(typeof(game)).")

observation_space(game::Kernel.AbstractGame, player::Int) =
    error("observation_space not implemented for $(typeof(game)).")

"""
Optional structured action-space declaration.

This is metadata / exact-layer functionality, not part of the kernel hot path.
For `SpaceActions` games, `Kernel.legal_actions(game, state, player)` should return
the current structured action domain directly.
"""
action_space(game::Kernel.AbstractGame, state, player::Int) =
    error("action_space not implemented for $(typeof(game)).")

"""
Canonical indexed base action space for IndexedActions games.
Only valid for games with `action_mode(typeof(game)) === IndexedActions`.
"""
base_action_space(game::Kernel.AbstractGame, player::Int) =
    Spaces.IndexedDiscreteSpace(Kernel.num_base_actions(game, player))

"""
Optional imperfect-information APIs.
"""
information_state(game::Kernel.AbstractGame, state, player::Int) =
    error("information_state not implemented for $(typeof(game)).")

public_observation(game::Kernel.AbstractGame, state) =
    error("public_observation not implemented for $(typeof(game)).")

public_state(game::Kernel.AbstractGame, state) =
    error("public_state not implemented for $(typeof(game)).")

"""
Optional transition-dependent observation API.

Relationship to the kernel observation contract:
- `Kernel.observe(game, next_state, player)` is the canonical default post-transition observation
- `observe_transition(...)` is the richer optional hook for games where the emitted observation depends
  on `(state, action, next_state)` rather than on `next_state` alone
"""
observe_transition(game::Kernel.AbstractGame, state, action, next_state, player::Int) =
    Kernel.observe(game, next_state, player)

"""
Optional exact stochastic APIs.
"""
chance_outcomes(game::Kernel.AbstractGame, state) =
    error("chance_outcomes not implemented for $(typeof(game)).")

transition_kernel(game::Kernel.AbstractGame, state, action) =
    error("transition_kernel not implemented for $(typeof(game)).")

observation_kernel(game::Kernel.AbstractGame, state, action, next_state) =
    error("observation_kernel not implemented for $(typeof(game)).")

"""
Optional exact solver/deviation API.
"""
counterfactual_values(game::Kernel.AbstractGame, state, player::Int, joint_action) =
    error("counterfactual_values not implemented for $(typeof(game)).")

"""
Optional exact terminal payoff hook.

This is intended for exact compiled solvers and analysis layers that need
terminal utilities without requiring an artificial action or transition call.

Contract:
- valid only when `Kernel.is_terminal(game, state)` is true
- returns the same reward container shape used by the game
"""
terminal_payoffs(game::Kernel.AbstractGame, state) =
    error("terminal_payoffs not implemented for $(typeof(game)).")

end