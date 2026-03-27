module DecisionRuleProfiles

using Random
using ..Kernel
using ..DecisionRulesInterface

export DecisionRuleProfile
export num_rules

export conditioning_kinds
export memory_classes

export is_unconditioned
export is_observation_conditioned
export is_state_conditioned
export is_history_conditioned
export is_infoset_conditioned
export is_memoryless

export require_profile_length
export require_unconditioned
export require_observation_conditioned
export require_state_conditioned
export require_history_conditioned
export require_infoset_conditioned
export require_memoryless

export unconditioned_rule_profile
export observation_rule_profile
export state_rule_profile
export history_rule_profile
export infoset_rule_profile

export local_observations
export sample_action_tuple
export sample_joint_action
export sample_profile_action
export joint_probability
export joint_density

"""
Tuple-backed fixed-player decision-rule profile.

The field type is left as a concrete tuple type to preserve heterogeneous
rule types for inference. Constructor validation enforces that all entries
are decision rules.
"""
struct DecisionRuleProfile{N,R<:Tuple}
    rules::R
end

function DecisionRuleProfile(rules::R) where {R<:Tuple}
    N = length(rules)
    @inbounds for i in 1:N
        rules[i] isa DecisionRulesInterface.AbstractDecisionRule ||
            throw(ArgumentError(
                "All entries of a DecisionRuleProfile must subtype AbstractDecisionRule. Entry $i has type $(typeof(rules[i]))."
            ))
    end
    return DecisionRuleProfile{N,R}(rules)
end

Base.getindex(p::DecisionRuleProfile, i::Int) = p.rules[i]
Base.length(::DecisionRuleProfile{N}) where {N} = N
Base.iterate(p::DecisionRuleProfile, st...) = iterate(p.rules, st...)
Base.eltype(::Type{<:DecisionRuleProfile}) = DecisionRulesInterface.AbstractDecisionRule
Base.firstindex(::DecisionRuleProfile) = 1
Base.lastindex(p::DecisionRuleProfile) = length(p)
Base.Tuple(p::DecisionRuleProfile) = p.rules

num_rules(p::DecisionRuleProfile{N}) where {N} = N
num_rules(p::Tuple) = length(p)

# ----------------------------------------------------------------------
# Introspection
# ----------------------------------------------------------------------

conditioning_kinds(profile::DecisionRuleProfile) =
    ntuple(i -> DecisionRulesInterface.conditioning_kind(profile[i]), length(profile))

memory_classes(profile::DecisionRuleProfile) =
    ntuple(i -> DecisionRulesInterface.memory_class(profile[i]), length(profile))

@inline function _all_conditioning(profile::DecisionRuleProfile, ::Type{K}) where {K<:DecisionRulesInterface.AbstractConditioningKind}
    @inbounds for i in eachindex(profile.rules)
        DecisionRulesInterface.conditioning_kind(profile[i]) isa K || return false
    end
    return true
end

is_unconditioned(profile::DecisionRuleProfile) =
    _all_conditioning(profile, DecisionRulesInterface.NoConditioning)

is_observation_conditioned(profile::DecisionRuleProfile) =
    _all_conditioning(profile, DecisionRulesInterface.ObservationConditioning)

is_state_conditioned(profile::DecisionRuleProfile) =
    _all_conditioning(profile, DecisionRulesInterface.StateConditioning)

is_history_conditioned(profile::DecisionRuleProfile) =
    _all_conditioning(profile, DecisionRulesInterface.HistoryConditioning)

is_infoset_conditioned(profile::DecisionRuleProfile) =
    _all_conditioning(profile, DecisionRulesInterface.InfosetConditioning)

function is_memoryless(profile::DecisionRuleProfile)
    @inbounds for i in eachindex(profile.rules)
        DecisionRulesInterface.memory_class(profile[i]) isa DecisionRulesInterface.Memoryless || return false
    end
    return true
end

# ----------------------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------------------

function require_profile_length(profile::DecisionRuleProfile, game::Kernel.AbstractGame)
    N = Kernel.num_players(game)
    length(profile) == N || throw(ArgumentError(
        "Decision-rule profile length $(length(profile)) does not match number of players $N."
    ))
    return profile
end

function require_unconditioned(profile::DecisionRuleProfile)
    is_unconditioned(profile) || throw(ArgumentError(
        "Expected an unconditioned decision-rule profile."
    ))
    return profile
end

function require_observation_conditioned(profile::DecisionRuleProfile)
    is_observation_conditioned(profile) || throw(ArgumentError(
        "Expected an observation-conditioned decision-rule profile."
    ))
    return profile
end

function require_state_conditioned(profile::DecisionRuleProfile)
    is_state_conditioned(profile) || throw(ArgumentError(
        "Expected a state-conditioned decision-rule profile."
    ))
    return profile
end

function require_history_conditioned(profile::DecisionRuleProfile)
    is_history_conditioned(profile) || throw(ArgumentError(
        "Expected a history-conditioned decision-rule profile."
    ))
    return profile
end

function require_infoset_conditioned(profile::DecisionRuleProfile)
    is_infoset_conditioned(profile) || throw(ArgumentError(
        "Expected an infoset-conditioned decision-rule profile."
    ))
    return profile
end

function require_memoryless(profile::DecisionRuleProfile)
    is_memoryless(profile) || throw(ArgumentError(
        "Expected a memoryless decision-rule profile."
    ))
    return profile
end

# ----------------------------------------------------------------------
# Convenience constructors
# ----------------------------------------------------------------------

unconditioned_rule_profile(rules::Tuple) =
    require_unconditioned(DecisionRuleProfile(rules))

observation_rule_profile(rules::Tuple) =
    require_observation_conditioned(DecisionRuleProfile(rules))

state_rule_profile(rules::Tuple) =
    require_state_conditioned(DecisionRuleProfile(rules))

history_rule_profile(rules::Tuple) =
    require_history_conditioned(DecisionRuleProfile(rules))

infoset_rule_profile(rules::Tuple) =
    require_infoset_conditioned(DecisionRuleProfile(rules))

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

@inline function _query_runtime_action(rule, game, state, p::Int, rng::AbstractRNG)
    kind = DecisionRulesInterface.conditioning_kind(rule)

    if kind isa DecisionRulesInterface.NoConditioning
        return DecisionRulesInterface.sample_action(rule, rng)

    elseif kind isa DecisionRulesInterface.ObservationConditioning
        obs = Kernel.observe(game, state, p)
        return DecisionRulesInterface.sample_action(rule, obs, rng)

    elseif kind isa DecisionRulesInterface.StateConditioning
        return DecisionRulesInterface.sample_action(rule, state, rng)

    else
        throw(ArgumentError(
            "Generic runtime profile sampling does not support $(typeof(kind)) rules. Use a domain-specific helper for that conditioning kind."
        ))
    end
end

# ----------------------------------------------------------------------
# Joint / profile sampling
# ----------------------------------------------------------------------

@inline function sample_action_tuple(profile::DecisionRuleProfile,
                                     rng::AbstractRNG = Random.default_rng())
    return ntuple(i -> DecisionRulesInterface.sample_action(profile[i], rng), length(profile))
end

@inline function _sample_joint_action_tuple(profile::Tuple, rng::AbstractRNG)
    acts = ntuple(i -> DecisionRulesInterface.sample_action(profile[i], rng), length(profile))
    return Kernel.joint_action(acts)
end

sample_joint_action(profile::DecisionRuleProfile,
                    rng::AbstractRNG = Random.default_rng()) =
    _sample_joint_action_tuple(profile.rules, rng)

sample_joint_action(profile::Tuple{Vararg{DecisionRulesInterface.AbstractDecisionRule,N}},
                    rng::AbstractRNG = Random.default_rng()) where {N} =
    _sample_joint_action_tuple(profile, rng)

function sample_joint_action(profile::Tuple, rng::AbstractRNG = Random.default_rng())
    return sample_joint_action(DecisionRuleProfile(profile), rng)
end

"""
Sample the current kernel action induced by a decision-rule profile at `(game, state)`.

Supported generic conditioning kinds:
- `NoConditioning`
- `ObservationConditioning`
- `StateConditioning`

Other kinds, such as infoset-conditioned or history-conditioned rules, should use
domain-specific helpers that know how to construct the relevant conditioning object.
"""
function sample_profile_action(profile::DecisionRuleProfile,
                               game::Kernel.AbstractGame,
                               state,
                               rng::AbstractRNG = Random.default_rng())
    require_profile_length(profile, game)

    nk = Kernel.node_kind(game, state)

    if nk == Kernel.DECISION
        p = Kernel.only_acting_player(game, state)
        a = _query_runtime_action(profile[p], game, state, p, rng)
        return _check_legal_action(game, state, p, a)

    elseif nk == Kernel.SIMULTANEOUS
        aps = Tuple(Kernel.acting_players(game, state))
        acts = ntuple(i -> begin
            p = aps[i]
            a = _query_runtime_action(profile[p], game, state, p, rng)
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

# ----------------------------------------------------------------------
# Joint / profile probabilities and densities
# ----------------------------------------------------------------------

@inline function _joint_probability_tuple(profile::Tuple, actions::Tuple)
    length(profile) == length(actions) ||
        throw(ArgumentError("Profile and action tuple must have the same length."))
    p = 1.0
    @inbounds for i in eachindex(profile)
        p *= DecisionRulesInterface.action_probability(profile[i], actions[i])
    end
    return p
end

joint_probability(profile::DecisionRuleProfile, actions::Tuple) =
    _joint_probability_tuple(profile.rules, actions)

joint_probability(profile::Tuple{Vararg{DecisionRulesInterface.AbstractDecisionRule,N}},
                  actions::Tuple) where {N} =
    _joint_probability_tuple(profile, actions)

function joint_probability(profile::Tuple, actions::Tuple)
    return joint_probability(DecisionRuleProfile(profile), actions)
end

@inline function _joint_density_tuple(profile::Tuple, actions::Tuple)
    length(profile) == length(actions) ||
        throw(ArgumentError("Profile and action tuple must have the same length."))
    d = 1.0
    @inbounds for i in eachindex(profile)
        d *= DecisionRulesInterface.action_density(profile[i], actions[i])
    end
    return d
end

joint_density(profile::DecisionRuleProfile, actions::Tuple) =
    _joint_density_tuple(profile.rules, actions)

joint_density(profile::Tuple{Vararg{DecisionRulesInterface.AbstractDecisionRule,N}},
              actions::Tuple) where {N} =
    _joint_density_tuple(profile, actions)

function joint_density(profile::Tuple, actions::Tuple)
    return joint_density(DecisionRuleProfile(profile), actions)
end

end