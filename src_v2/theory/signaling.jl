module Signaling

using Random
using ..Strategies
using ..Bayesian
using ..Classification

export SenderReceiverRoles
export SignalingProfile
export sender_strategy, receiver_strategy
export sample_message, sample_receiver_action
export induced_message_distribution

struct SenderReceiverRoles
    sender::Int
    receiver::Int
end

struct SignalingProfile{S,R}
    sender::S
    receiver::R
end

sender_strategy(p::SignalingProfile) = p.sender
receiver_strategy(p::SignalingProfile) = p.receiver

sample_message(p::SignalingProfile, sender_type, rng::AbstractRNG = Random.default_rng()) =
    Strategies.sample_action(p.sender, sender_type, rng)

sample_receiver_action(p::SignalingProfile, message, rng::AbstractRNG = Random.default_rng()) =
    Strategies.sample_action(p.receiver, message, rng)

function induced_message_distribution(prior, sender_behavior, sender_player::Int)
    T = Bayesian.type_space(prior, sender_player)
    types = T.elements
    length(types) > 0 || throw(ArgumentError("Sender type space must be nonempty."))

    first_ls = Strategies.local_strategy(sender_behavior, types[first(eachindex(types))])
    msgs = Strategies.support(first_ls)
    M = typeof(msgs[first(eachindex(msgs))])

    msg_prob = Dict{M,Float64}()

    for t in types
        pt = Bayesian.marginal_probability(prior, sender_player, t)
        ls = Strategies.local_strategy(sender_behavior, t)
        A = Strategies.support(ls)
        P = Strategies.probabilities(ls)
        @inbounds for i in eachindex(P)
            msg_prob[A[i]] = get(msg_prob, A[i], 0.0) + pt * P[i]
        end
    end

    msg_keys = Tuple(keys(msg_prob))
    msg_vals = ntuple(i -> msg_prob[msg_keys[i]], length(msg_keys))
    return Strategies.FiniteMixedStrategy(msg_keys, msg_vals)
end

Classification.is_signaling_game(::SignalingProfile) = true

end