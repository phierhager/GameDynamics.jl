module RuleExecution

using Random

using ..Kernel
using ..DecisionRulesInterface
using ..DecisionRuleProfiles
using ..ExtensiveFormInfosets

export require_profile_length
export player_context
export sample_rule_action
export action_probability_at
export sample_profile_action

"""
Check that a decision-rule profile matches the player count of a game.

This lives in runtime compatibility code, not in `decision_rules/profiles.jl`,
because it is a profile-world compatibility check rather than an intrinsic
profile property.
"""
function require_profile_length(profile::DecisionRuleProfiles.DecisionRuleProfile,
                                game::Kernel.AbstractGame)
    N = Kernel.num_players(game)
    length(profile) == N || throw(ArgumentError(
        "Decision-rule profile length $(length(profile)) does not match number of players $N."
    ))
    return profile
end

"""
Construct the context object that `rule` should see for `player` at `(game, state)`.

Supported built-in context kinds:
- `NoContext`
- `ObservationContext`
- `StateContext`
- `InfosetContext`

`HistoryContext` and `CustomContext` are intentionally not handled generically here,
because they require domain-specific context construction.
"""
function player_context(rule::DecisionRulesInterface.AbstractDecisionRule,
                        game::Kernel.AbstractGame,
                        state,
                        player::Int)
    kind = DecisionRulesInterface.context_kind(rule)

    if kind isa DecisionRulesInterface.NoContext
        return nothing

    elseif kind isa DecisionRulesInterface.ObservationContext
        return Kernel.observe(game, state, player)

    elseif kind isa DecisionRulesInterface.StateContext
        return state

    elseif kind isa DecisionRulesInterface.InfosetContext
        return ExtensiveFormInfosets.infoset(game, state, player)

    elseif kind isa DecisionRulesInterface.HistoryContext
        throw(ArgumentError(
            "Generic runtime querying cannot construct a history context. Use a domain-specific helper."
        ))

    else
        throw(ArgumentError(
            "Generic runtime querying does not know how to construct context kind $(typeof(kind))."
        ))
    end
end

"""
Sample the action induced by `rule` for `player` at `(game, state)`.
"""
function sample_rule_action(rule::DecisionRulesInterface.AbstractDecisionRule,
                            game::Kernel.AbstractGame,
                            state,
                            player::Int,
                            rng::AbstractRNG = Random.default_rng())
    ctx = player_context(rule, game, state, player)
    if ctx === nothing
        return DecisionRulesInterface.sample_action(rule, rng)
    else
        return DecisionRulesInterface.sample_action(rule, ctx, rng)
    end
end

"""
Return the action probability induced by `rule` for `player` at `(game, state)`.
"""
function action_probability_at(rule::DecisionRulesInterface.AbstractDecisionRule,
                               game::Kernel.AbstractGame,
                               state,
                               player::Int,
                               action)
    ctx = player_context(rule, game, state, player)
    if ctx === nothing
        return DecisionRulesInterface.action_probability(rule, action)
    else
        return DecisionRulesInterface.action_probability(rule, ctx, action)
    end
end

@inline function _check_legal_action(game, state, p::Int, a)
    legal = Kernel.legal_actions(game, state, p)
    a in legal || throw(ArgumentError("Decision rule produced illegal action $a for player $p."))
    return a
end

"""
Sample the current kernel action induced by a decision-rule profile at `(game, state)`.

This is the runtime bridge from:
- per-player decision rules
- world semantics and information structure
to a kernel action usable by `Kernel.step`.
"""
function sample_profile_action(profile::DecisionRuleProfiles.DecisionRuleProfile,
                               game::Kernel.AbstractGame,
                               state,
                               rng::AbstractRNG = Random.default_rng())
    require_profile_length(profile, game)

    nk = Kernel.node_kind(game, state)

    if nk == Kernel.DECISION
        p = Kernel.only_acting_player(game, state)
        a = sample_rule_action(profile[p], game, state, p, rng)
        return _check_legal_action(game, state, p, a)

    elseif nk == Kernel.SIMULTANEOUS
        aps = Tuple(Kernel.acting_players(game, state))
        acts = ntuple(i -> begin
            p = aps[i]
            a = sample_rule_action(profile[p], game, state, p, rng)
            _check_legal_action(game, state, p, a)
        end, length(aps))
        ja = Kernel.JointAction(acts)
        return Kernel.validate_joint_action(game, state, ja)

    elseif nk == Kernel.CHANCE
        return Kernel.SampleChance()

    else
        throw(ArgumentError("Cannot sample an action at a terminal node."))
    end
end

function sample_profile_action(profile::Tuple,
                               game::Kernel.AbstractGame,
                               state,
                               rng::AbstractRNG = Random.default_rng())
    return sample_profile_action(
        DecisionRuleProfiles.DecisionRuleProfile(profile),
        game,
        state,
        rng,
    )
end

end