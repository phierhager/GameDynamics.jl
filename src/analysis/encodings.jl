module Encodings

import Base: sizehint!

export DenseEncoder
export IdentityIntEncoder
export DenseIntRangeEncoder
export InfosetEncoder, ActionEncoder, ObservationEncoder, TypeProfileEncoder
export encode!, encode, decode, has_encoding, sizehint!, reset!

mutable struct DenseEncoder{K}
    to_id::Dict{K,Int}
    from_id::Vector{K}
end

DenseEncoder{K}() where {K} = DenseEncoder(Dict{K,Int}(), K[])

function sizehint!(enc::DenseEncoder, n::Int)
    Base.sizehint!(enc.to_id, n)
    Base.sizehint!(enc.from_id, n)
    return enc
end

has_encoding(enc::DenseEncoder, x) = haskey(enc.to_id, x)

function encode!(enc::DenseEncoder{K}, x::K) where {K}
    get!(enc.to_id, x) do
        push!(enc.from_id, x)
        return length(enc.from_id)
    end
end

function encode(enc::DenseEncoder{K}, x::K) where {K}
    haskey(enc.to_id, x) || throw(KeyError("Value not encoded: $x"))
    return enc.to_id[x]
end

function decode(enc::DenseEncoder, i::Int)
    1 <= i <= length(enc.from_id) || throw(BoundsError(enc.from_id, i))
    return enc.from_id[i]
end

function reset!(enc::DenseEncoder)
    empty!(enc.to_id)
    empty!(enc.from_id)
    return enc
end

"""
Fast path for already-dense integer ids: encode/decode are identity.
"""
struct IdentityIntEncoder end

sizehint!(::IdentityIntEncoder, n::Int) = nothing
has_encoding(::IdentityIntEncoder, x::Int) = x >= 1
encode!(::IdentityIntEncoder, x::Int) = x
encode(::IdentityIntEncoder, x::Int) = x
decode(::IdentityIntEncoder, i::Int) = i
reset!(::IdentityIntEncoder) = nothing

"""
Fast path when ids are dense integers on a fixed shifted range:
external ids in [offset+1, offset+n] map to internal ids 1:n.
"""
struct DenseIntRangeEncoder
    offset::Int
    n::Int
end

has_encoding(enc::DenseIntRangeEncoder, x::Int) = enc.offset + 1 <= x <= enc.offset + enc.n
function encode!(enc::DenseIntRangeEncoder, x::Int)
    has_encoding(enc, x) || throw(KeyError("Out of range: $x"))
    return x - enc.offset
end
encode(enc::DenseIntRangeEncoder, x::Int) = encode!(enc, x)
function decode(enc::DenseIntRangeEncoder, i::Int)
    1 <= i <= enc.n || throw(BoundsError(1:enc.n, i))
    return i + enc.offset
end
sizehint!(::DenseIntRangeEncoder, n::Int) = nothing
reset!(enc::DenseIntRangeEncoder) = enc

const InfosetEncoder = DenseEncoder
const ActionEncoder = DenseEncoder
const ObservationEncoder = DenseEncoder
const TypeProfileEncoder = DenseEncoder

end