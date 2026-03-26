module PolicyProfiles

using Random
using ..Kernel
using ..PolicyCore
using ..POSG

export ObservationPolicyProfile
export UnconditionedPolicyProfile
export StatePolicyProfile
export POSGProfile
export DecPOMDPProfile

export local_observations
export sample_profile_action

"""
Lower-level utility profile for observation-conditioned policies.
"""
struct ObservationPolicyProfile{P<:Tuple}
    policies::P
end

"""
Noncanonical runtime profile for unconditioned policies.
"""
struct UnconditionedPolicyProfile{P<:Tuple}
    policies::P
end

"""
Canonical runtime profile for state-conditioned policies.
"""
struct StatePolicyProfile{P<:Tuple}
    policies::P
end

"""
Canonical observation-based memoryless POSG runtime profile.
"""
struct POSGProfile{P<:Tuple}
    policies::P
end

"""
Canonical observation-based memoryless DecPOMDP runtime profile.
"""
struct DecPOMDPProfile{P<:Tuple}
    policies::P
end

# ----------------------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------------------

@inline function _validate_profile_length(policies::Tuple, game::Kernel.AbstractGame)
    N = Kernel.num_players(game)
    length(policies) == N || throw(ArgumentError(
        "Policy profile length $(length(policies)) does not match number of players $N."
    ))
    return N
end

@inline function _require_policies(policies::Tuple)
    @inbounds for i in eachindex(policies)
        policies[i] isa PolicyCore.AbstractPolicy ||
            throw(ArgumentError(
                "All entries must subtype AbstractPolicy. Entry $i has type $(typeof(policies[i]))."
            ))
    end
    return nothing
end

@inline function _check_policy_kinds_obs(policies::Tuple)
    @inbounds for i in eachindex(policies)
        p = policies[i]
        PolicyCore.conditioning_kind(p) isa PolicyCore.ObservationConditioning ||
            throw(ArgumentError("Expected observation-conditioned policy at slot $i."))
    end
    return nothing
end

@inline function _check_policy_kinds_state(policies::Tuple)
    @inbounds for i in eachindex(policies)
        p = policies[i]
        PolicyCore.conditioning_kind(p) isa PolicyCore.StateConditioning ||
            throw(ArgumentError("Expected state-conditioned policy at slot $i."))
    end
    return nothing
end

@inline function _check_policy_kinds_none(policies::Tuple)
    @inbounds for i in eachindex(policies)
        p = policies[i]
        PolicyCore.conditioning_kind(p) isa PolicyCore.NoConditioning ||
            throw(ArgumentError("Expected unconditioned policy at slot $i."))
    end
    return nothing
end

@inline function _check_memoryless(policies::Tuple, who::AbstractString)
    @inbounds for i in eachindex(policies)
        PolicyCore.memory_class(policies[i]) isa PolicyCore.Memoryless ||
            throw(ArgumentError(
                "$who requires memoryless local policies; slot $i is $(typeof(PolicyCore.memory_class(policies[i])))."
            ))
    end
    return nothing
end

# ----------------------------------------------------------------------
# Constructors
# ----------------------------------------------------------------------

function ObservationPolicyProfile(policies::P) where {P<:Tuple}
    _require_policies(policies)
    _check_policy_kinds_obs(policies)
    return ObservationPolicyProfile{P}(policies)
end

function UnconditionedPolicyProfile(policies::P) where {P<:Tuple}
    _require_policies(policies)
    _check_policy_kinds_none(policies)
    return UnconditionedPolicyProfile{P}(policies)
end

function StatePolicyProfile(policies::P) where {P<:Tuple}
    _require_policies(policies)
    _check_policy_kinds_state(policies)
    return StatePolicyProfile{P}(policies)
end

function POSGProfile(policies::P) where {P<:Tuple}
    _require_policies(policies)
    _check_policy_kinds_obs(policies)
    _check_memoryless(policies, "Canonical POSGProfile")
    return POSGProfile{P}(policies)
end

function DecPOMDPProfile(policies::P) where {P<:Tuple}
    _require_policies(policies)
    _check_policy_kinds_obs(policies)
    _check_memoryless(policies, "Canonical DecPOMDPProfile")
    return DecPOMDPProfile{P}(policies)
end

# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

local_observations(game::Kernel.AbstractGame, state) =
    ntuple(i -> Kernel.observe(game, state, i), Kernel.num_players(game))

@inline function _check_legal_action(game, state, p::Int, a)
    legal = Kernel.legal_actions(game, state, p)
    a in legal || throw(ArgumentError("Policy produced illegal action $a for player $p."))
    return a
end

# ----------------------------------------------------------------------
# Sampling helpers
# ----------------------------------------------------------------------

@inline only_acting_player(game::Kernel.AbstractGame, state) = only(Kernel.acting_players(game, state))

function sample_profile_action(profile::UnconditionedPolicyProfile,
                               game::Kernel.AbstractGame,
                               state,
                               rng::AbstractRNG = Random.default_rng())
    _validate_profile_length(profile.policies, game)

    nk = Kernel.node_kind(game, state)
    if nk == Kernel.DECISION
        p = only_acting_player(game, state)
        a = PolicyCore.sample_action(profile.policies[p], rng)
        return _check_legal_action(game, state, p, a)

    elseif nk == Kernel.SIMULTANEOUS
        aps = Tuple(Kernel.acting_players(game, state))
        acts = ntuple(i -> begin
            p = aps[i]
            a = PolicyCore.sample_action(profile.policies[p], rng)
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

function sample_profile_action(profile::ObservationPolicyProfile,
                               game::Kernel.AbstractGame,
                               state,
                               rng::AbstractRNG = Random.default_rng())
    _validate_profile_length(profile.policies, game)

    nk = Kernel.node_kind(game, state)
    if nk == Kernel.DECISION
        p = only_acting_player(game, state)
        observation = Kernel.observe(game, state, p)
        a = PolicyCore.sample_action(profile.policies[p], observation, rng)
        return _check_legal_action(game, state, p, a)

    elseif nk == Kernel.SIMULTANEOUS
        aps = Tuple(Kernel.acting_players(game, state))
        acts = ntuple(i -> begin
            p = aps[i]
            observation = Kernel.observe(game, state, p)
            a = PolicyCore.sample_action(profile.policies[p], observation, rng)
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

function sample_profile_action(profile::StatePolicyProfile,
                               game::Kernel.AbstractGame,
                               state,
                               rng::AbstractRNG = Random.default_rng())
    _validate_profile_length(profile.policies, game)

    nk = Kernel.node_kind(game, state)
    if nk == Kernel.DECISION
        p = only_acting_player(game, state)
        a = PolicyCore.sample_action(profile.policies[p], state, rng)
        return _check_legal_action(game, state, p, a)

    elseif nk == Kernel.SIMULTANEOUS
        aps = Tuple(Kernel.acting_players(game, state))
        acts = ntuple(i -> begin
            p = aps[i]
            a = PolicyCore.sample_action(profile.policies[p], state, rng)
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

function sample_profile_action(profile::POSGProfile,
                               game::Kernel.AbstractGame,
                               state,
                               rng::AbstractRNG = Random.default_rng())
    POSG.require_valid_posg(game)
    return sample_profile_action(ObservationPolicyProfile(profile.policies), game, state, rng)
end

function sample_profile_action(profile::DecPOMDPProfile,
                               game::Kernel.AbstractGame,
                               state,
                               rng::AbstractRNG = Random.default_rng())
    POSG.require_valid_decpomdp(game)
    return sample_profile_action(ObservationPolicyProfile(profile.policies), game, state, rng)
end

end