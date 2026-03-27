module DecisionRuleProfiles

using Random
using ..Kernel
using ..DecisionRulesInterface
using ..POSG

export ObservationRuleProfile
export UnconditionedRuleProfile
export StateRuleProfile
export POSGRuleProfile
export DecPOMDPRuleProfile

export local_observations
export sample_profile_action

"""
Lower-level utility profile for observation-conditioned decision rules.
"""
struct ObservationRuleProfile{P<:Tuple}
    rules::P
end

"""
Noncanonical runtime profile for unconditioned decision rules.
"""
struct UnconditionedRuleProfile{P<:Tuple}
    rules::P
end

"""
Canonical runtime profile for state-conditioned decision rules.
"""
struct StateRuleProfile{P<:Tuple}
    rules::P
end

"""
Canonical observation-based memoryless POSG runtime profile.
"""
struct POSGRuleProfile{P<:Tuple}
    rules::P
end

"""
Canonical observation-based memoryless Dec-POMDP runtime profile.
"""
struct DecPOMDPRuleProfile{P<:Tuple}
    rules::P
end

# ----------------------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------------------

@inline function _validate_profile_length(rules::Tuple, game::Kernel.AbstractGame)
    N = Kernel.num_players(game)
    length(rules) == N || throw(ArgumentError(
        "Decision-rule profile length $(length(rules)) does not match number of players $N."
    ))
    return N
end

@inline function _require_rules(rules::Tuple)
    @inbounds for i in eachindex(rules)
        rules[i] isa DecisionRulesInterface.AbstractDecisionRule ||
            throw(ArgumentError(
                "All entries must subtype AbstractDecisionRule. Entry $i has type $(typeof(rules[i]))."
            ))
    end
    return nothing
end

@inline function _check_rule_kinds_obs(rules::Tuple)
    @inbounds for i in eachindex(rules)
        r = rules[i]
        DecisionRulesInterface.conditioning_kind(r) isa DecisionRulesInterface.ObservationConditioning ||
            throw(ArgumentError("Expected observation-conditioned decision rule at slot $i."))
    end
    return nothing
end

@inline function _check_rule_kinds_state(rules::Tuple)
    @inbounds for i in eachindex(rules)
        r = rules[i]
        DecisionRulesInterface.conditioning_kind(r) isa DecisionRulesInterface.StateConditioning ||
            throw(ArgumentError("Expected state-conditioned decision rule at slot $i."))
    end
    return nothing
end

@inline function _check_rule_kinds_none(rules::Tuple)
    @inbounds for i in eachindex(rules)
        r = rules[i]
        DecisionRulesInterface.conditioning_kind(r) isa DecisionRulesInterface.NoConditioning ||
            throw(ArgumentError("Expected unconditioned decision rule at slot $i."))
    end
    return nothing
end

@inline function _check_memoryless(rules::Tuple, who::AbstractString)
    @inbounds for i in eachindex(rules)
        DecisionRulesInterface.memory_class(rules[i]) isa DecisionRulesInterface.Memoryless ||
            throw(ArgumentError(
                "$who requires memoryless local decision rules; slot $i is $(typeof(DecisionRulesInterface.memory_class(rules[i])))."
            ))
    end
    return nothing
end

# ----------------------------------------------------------------------
# Constructors
# ----------------------------------------------------------------------

function ObservationRuleProfile(rules::P) where {P<:Tuple}
    _require_rules(rules)
    _check_rule_kinds_obs(rules)
    return ObservationRuleProfile{P}(rules)
end

function UnconditionedRuleProfile(rules::P) where {P<:Tuple}
    _require_rules(rules)
    _check_rule_kinds_none(rules)
    return UnconditionedRuleProfile{P}(rules)
end

function StateRuleProfile(rules::P) where {P<:Tuple}
    _require_rules(rules)
    _check_rule_kinds_state(rules)
    return StateRuleProfile{P}(rules)
end

function POSGRuleProfile(rules::P) where {P<:Tuple}
    _require_rules(rules)
    _check_rule_kinds_obs(rules)
    _check_memoryless(rules, "Canonical POSGRuleProfile")
    return POSGRuleProfile{P}(rules)
end

function DecPOMDPRuleProfile(rules::P) where {P<:Tuple}
    _require_rules(rules)
    _check_rule_kinds_obs(rules)
    _check_memoryless(rules, "Canonical DecPOMDPRuleProfile")
    return DecPOMDPRuleProfile{P}(rules)
end

# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

local_observations(game::Kernel.AbstractGame, state) =
    ntuple(i -> Kernel.observe(game, state, i), Kernel.num_players(game))

@inline function _check_legal_action(game, state, p::Int, a)
    legal = Kernel.legal_actions(game, state, p)
    a in legal || throw(ArgumentError("Decision rule produced illegal action $a for player $p."))
    return a
end

# ----------------------------------------------------------------------
# Sampling helpers
# ----------------------------------------------------------------------

@inline function _sample_profile_action_impl(rules::Tuple,
                                             game::Kernel.AbstractGame,
                                             state,
                                             rng::AbstractRNG,
                                             query_action)
    _validate_profile_length(rules, game)

    nk = Kernel.node_kind(game, state)

    if nk == Kernel.DECISION
        p = Kernel.only_acting_player(game, state)
        a = query_action(p, state, rng)
        return _check_legal_action(game, state, p, a)

    elseif nk == Kernel.SIMULTANEOUS
        aps = Tuple(Kernel.acting_players(game, state))
        acts = ntuple(i -> begin
            p = aps[i]
            a = query_action(p, state, rng)
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

function sample_profile_action(profile::UnconditionedRuleProfile,
                               game::Kernel.AbstractGame,
                               state,
                               rng::AbstractRNG = Random.default_rng())
    return _sample_profile_action_impl(profile.rules, game, state, rng) do p, st, rr
        DecisionRulesInterface.sample_action(profile.rules[p], rr)
    end
end

function sample_profile_action(profile::ObservationRuleProfile,
                               game::Kernel.AbstractGame,
                               state,
                               rng::AbstractRNG = Random.default_rng())
    return _sample_profile_action_impl(profile.rules, game, state, rng) do p, st, rr
        obs = Kernel.observe(game, st, p)
        DecisionRulesInterface.sample_action(profile.rules[p], obs, rr)
    end
end

function sample_profile_action(profile::StateRuleProfile,
                               game::Kernel.AbstractGame,
                               state,
                               rng::AbstractRNG = Random.default_rng())
    return _sample_profile_action_impl(profile.rules, game, state, rng) do p, st, rr
        DecisionRulesInterface.sample_action(profile.rules[p], st, rr)
    end
end

function sample_profile_action(profile::POSGRuleProfile,
                               game::Kernel.AbstractGame,
                               state,
                               rng::AbstractRNG = Random.default_rng())
    POSG.require_valid_posg(game)
    return _sample_profile_action_impl(profile.rules, game, state, rng) do p, st, rr
        obs = Kernel.observe(game, st, p)
        DecisionRulesInterface.sample_action(profile.rules[p], obs, rr)
    end
end

function sample_profile_action(profile::DecPOMDPRuleProfile,
                               game::Kernel.AbstractGame,
                               state,
                               rng::AbstractRNG = Random.default_rng())
    POSG.require_valid_decpomdp(game)
    return _sample_profile_action_impl(profile.rules, game, state, rng) do p, st, rr
        obs = Kernel.observe(game, st, p)
        DecisionRulesInterface.sample_action(profile.rules[p], obs, rr)
    end
end

end