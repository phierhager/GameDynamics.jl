module SignalingGames

using Random

using ..StrategyInterface
using ..LocalStrategies
using ..BayesianPriors

export SenderReceiverRoles
export SignalingProfile
export sender
export receiver
export sender_strategy
export receiver_strategy
export sample_message
export sample_receiver_action
export induced_message_distribution

struct SenderReceiverRoles
    sender::Int
    receiver::Int

    function SenderReceiverRoles(sender::Int, receiver::Int)
        sender > 0 || throw(ArgumentError("sender must be positive."))
        receiver > 0 || throw(ArgumentError("receiver must be positive."))
        sender != receiver || throw(ArgumentError("sender and receiver must differ."))
        new(sender, receiver)
    end
end

sender(r::SenderReceiverRoles) = r.sender
receiver(r::SenderReceiverRoles) = r.receiver

struct SignalingProfile{S,R}
    sender::S
    receiver::R
end

sender_strategy(p::SignalingProfile) = p.sender
receiver_strategy(p::SignalingProfile) = p.receiver

function sample_message(p::SignalingProfile, sender_type, rng::AbstractRNG = Random.default_rng())
    return StrategyInterface.sample_action(p.sender, sender_type, rng)
end

function sample_receiver_action(p::SignalingProfile, message, rng::AbstractRNG = Random.default_rng())
    return StrategyInterface.sample_action(p.receiver, message, rng)
end

function induced_message_distribution(prior::BayesianPriors.AbstractPrior,
                                      sender_behavior,
                                      sender_player::Int)
    sender_player > 0 || throw(ArgumentError("sender_player must be positive."))

    T = BayesianPriors.type_space(prior, sender_player)
    hasproperty(T, :elements) || throw(ArgumentError(
        "induced_message_distribution requires a finite sender type space."
    ))

    types = T.elements
    length(types) > 0 || throw(ArgumentError("Sender type space must be nonempty."))

    first_local = sender_behavior isa StrategyInterface.AbstractRecordStrategy ?
        sender_behavior.extractor === nothing ? nothing : nothing : nothing

    first_rule = sender_behavior isa StrategyInterface.AbstractRecordStrategy ?
        nothing : nothing

    local0 = sender_behavior(types[1])
    local0 isa StrategyInterface.AbstractLocalStrategy || throw(ArgumentError(
        "sender_behavior(type) must return an AbstractLocalStrategy."
    ))

    A0 = StrategyInterface.support(local0)
    isempty(A0) && throw(ArgumentError("Sender local strategy support must be nonempty."))

    M = typeof(A0[1])
    @inbounds for i in 2:length(types)
        ls = sender_behavior(types[i])
        ls isa StrategyInterface.AbstractLocalStrategy || throw(ArgumentError(
            "sender_behavior(type) must return an AbstractLocalStrategy."
        ))
        A = StrategyInterface.support(ls)
        isempty(A) && throw(ArgumentError("Sender local strategy support must be nonempty."))
        M = typejoin(M, typeof(A[1]))
    end

    msg_prob = Dict{M,Float64}()
    order = Vector{M}()

    @inbounds for i in eachindex(types)
        t = types[i]
        pt = BayesianPriors.marginal_probability(prior, sender_player, t)

        ls = sender_behavior(t)
        A = StrategyInterface.support(ls)
        P = StrategyInterface.probabilities(ls)

        length(A) == length(P) || throw(ArgumentError(
            "Sender local support/probability lengths do not match for type $t."
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

    msgs = Tuple(order)
    probs = ntuple(i -> msg_prob[order[i]], length(order))
    return LocalStrategies.FiniteMixedStrategy(msgs, probs)
end

end