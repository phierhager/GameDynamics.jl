module Signaling

using Random
using ..DecisionRulesInterface
using ..DirectDecisionRules
using ..Bayesian
using ..Classification
using ..Spaces

export SenderReceiverRoles
export SignalingProfile
export sender_strategy, receiver_strategy
export sender, receiver
export sample_message, sample_receiver_action
export induced_message_distribution

"""
Sender/receiver role assignment for a signaling game.
"""
struct SenderReceiverRoles
    sender::Int
    receiver::Int

    function SenderReceiverRoles(sender::Int, receiver::Int)
        sender > 0 || throw(ArgumentError("sender must be positive, got $sender."))
        receiver > 0 || throw(ArgumentError("receiver must be positive, got $receiver."))
        sender != receiver || throw(ArgumentError("sender and receiver must differ."))
        new(sender, receiver)
    end
end

@inline sender(r::SenderReceiverRoles) = r.sender
@inline receiver(r::SenderReceiverRoles) = r.receiver

"""
Theory-level signaling profile.

This is intentionally a semantic/theory object, not a runtime `Policy` object.
Typical interpretation:
- `sender` is a type-conditioned signaling/behavior strategy
- `receiver` is a message-conditioned response strategy
"""
struct SignalingProfile{S,R}
    sender::S
    receiver::R
end

@inline sender_strategy(p::SignalingProfile) = p.sender
@inline receiver_strategy(p::SignalingProfile) = p.receiver

@inline function sample_message(
    p::SignalingProfile,
    sender_type,
    rng::AbstractRNG = Random.default_rng(),
)
    rule = p.sender
    if DecisionRulesInterface.context_kind(rule) isa DecisionRulesInterface.NoContext
        return DecisionRulesInterface.sample_action(rule, rng)
    else
        return DecisionRulesInterface.sample_action(rule, sender_type, rng)
    end
end

@inline function sample_receiver_action(
    p::SignalingProfile,
    message,
    rng::AbstractRNG = Random.default_rng(),
)
    rule = p.receiver
    if DecisionRulesInterface.context_kind(rule) isa DecisionRulesInterface.NoContext
        return DecisionRulesInterface.sample_action(rule, rng)
    else
        return DecisionRulesInterface.sample_action(rule, message, rng)
    end
end

# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

@inline function _support_eltype(s)
    return eltype(DecisionRulesInterface.support(s))
end

function _message_type(types, sender_behavior)
    first_type = first(types)
    first_ls = DecisionRulesInterface.local_rule(sender_behavior, first_type)
    first_support = DecisionRulesInterface.support(first_ls)
    isempty(first_support) && throw(ArgumentError(
        "Sender local strategy must have nonempty support for sender type $first_type."
    ))

    M = _support_eltype(first_ls)

    @inbounds for i in 2:length(types)
        t = types[i]
        ls = DecisionRulesInterface.local_rule(sender_behavior, t)
        A = DecisionRulesInterface.support(ls)
        isempty(A) && throw(ArgumentError(
            "Sender local strategy must have nonempty support for sender type $t."
        ))
        M = typejoin(M, _support_eltype(ls))
    end

    return M
end

"""
Compute the sender-induced marginal message distribution under a prior.

Requirements:
- the sender type space induced by the prior must be finite
- the sender behavior must provide a local strategy for each type
- each local strategy must expose finite support and probabilities

The returned object is a `FiniteMixedStrategy` over messages.

Implementation notes:
- message support is the union over all sender types
- support order is first-seen order across types, not string-sorted
- accumulation uses a typed dictionary keyed by the inferred message type
"""
function induced_message_distribution(prior, sender_behavior, sender_player::Int)
    sender_player > 0 || throw(ArgumentError("sender_player must be positive, got $sender_player."))

    T = Bayesian.type_space(prior, sender_player)
    T isa Spaces.FiniteSpace ||
        throw(ArgumentError("induced_message_distribution requires a finite sender type space."))

    types = T.elements
    length(types) > 0 || throw(ArgumentError("Sender type space must be nonempty."))

    M = _message_type(types, sender_behavior)

    msg_prob = Dict{M,Float64}()
    order = Vector{M}()
    sizehint!(order, 8)

    @inbounds for i in eachindex(types)
        t = types[i]
        pt = Bayesian.marginal_probability(prior, sender_player, t)
        ls = DecisionRulesInterface.local_rule(sender_behavior, t)
        A = DecisionRulesInterface.support(ls)
        P = DecisionRulesInterface.probabilities(ls)

        length(A) == length(P) || throw(ArgumentError(
            "Local sender strategy support/probability lengths do not match for sender type $t."
        ))

        for j in eachindex(P)
            m = convert(M, A[j])
            if !haskey(msg_prob, m)
                push!(order, m)
                msg_prob[m] = 0.0
            end
            msg_prob[m] += pt * Float64(P[j])
        end
    end

    isempty(order) && throw(ArgumentError(
        "Induced message distribution is empty; sender local decision rule must have nonempty support."
    ))

    msg_keys = Tuple(order)
    msg_vals = ntuple(i -> msg_prob[order[i]], length(order))

    return DirectDecisionRules.FiniteMixedDecisionRule(msg_keys, msg_vals)
end

Classification.is_signaling_game(::SignalingProfile) = true

end