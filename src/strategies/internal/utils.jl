module StrategyInternalUtils

export normalize_probs
export canonicalize_support_probs
export canonicalize_joint_tuple_probs

function normalize_probs(probs::Tuple)
    length(probs) > 0 || throw(ArgumentError("Probability tuple must be nonempty."))
    any(p -> p < 0, probs) && throw(ArgumentError("Probabilities must be nonnegative."))
    z = sum(probs)
    z > 0 || throw(ArgumentError("Probabilities must sum to a positive value."))
    return ntuple(i -> Float64(probs[i]) / z, length(probs))
end

function normalize_probs(probs::AbstractVector)
    isempty(probs) && throw(ArgumentError("Probability vector must be nonempty."))
    any(p -> p < 0, probs) && throw(ArgumentError("Probabilities must be nonnegative."))
    z = sum(probs)
    z > 0 || throw(ArgumentError("Probabilities must sum to a positive value."))
    return Float64.(probs) ./ z
end

function canonicalize_support_probs(actions::A, probs) where {A}
    length(actions) == length(probs) ||
        throw(ArgumentError("Actions and probabilities must have the same length."))
    isempty(actions) && throw(ArgumentError("Support must be nonempty."))

    p = normalize_probs(probs)

    T = eltype(actions)
    acc = Dict{T,Float64}()
    order = Vector{T}()

    @inbounds for i in eachindex(actions)
        a = actions[i]
        if !haskey(acc, a)
            push!(order, a)
            acc[a] = 0.0
        end
        acc[a] += p[i]
    end

    acts = Tuple(order)
    ps = ntuple(i -> acc[order[i]], length(order))
    return acts, ps
end

function canonicalize_joint_tuple_probs(joint_support::S, probs) where {S}
    length(joint_support) == length(probs) ||
        throw(ArgumentError("Joint support and probabilities must have the same length."))
    isempty(joint_support) && throw(ArgumentError("Joint support must be nonempty."))

    p = normalize_probs(probs)

    T = eltype(joint_support)
    acc = Dict{T,Float64}()
    order = Vector{T}()

    @inbounds for i in eachindex(joint_support)
        jt = joint_support[i]
        if !haskey(acc, jt)
            push!(order, jt)
            acc[jt] = 0.0
        end
        acc[jt] += p[i]
    end

    tuples = Tuple(order)
    ps = ntuple(i -> acc[order[i]], length(order))
    return tuples, ps
end

end