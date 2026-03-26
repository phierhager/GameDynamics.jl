module Spec

using ..Kernel

export HorizonKind, EPISODIC, CONTINUING
export PayoffKind, ZERO_SUM, GENERAL_SUM, CONSTANT_SUM, UNKNOWN_PAYOFF
export GameSpec, game_spec

@enum HorizonKind::UInt8 begin
    EPISODIC   = 0x01
    CONTINUING = 0x02
end

@enum PayoffKind::UInt8 begin
    ZERO_SUM       = 0x01
    GENERAL_SUM    = 0x02
    CONSTANT_SUM   = 0x03
    UNKNOWN_PAYOFF = 0x04
end

"""
Lightweight core metadata.

This is intentionally small:
- runtime may use `max_steps`
- theory/classification may use the semantic booleans when available
- no redundant `zero_sum` / `general_sum` booleans; use `payoff_kind`
"""
Base.@kwdef struct GameSpec
    horizon_kind::HorizonKind = EPISODIC
    payoff_kind::PayoffKind = UNKNOWN_PAYOFF

    max_steps::Union{Nothing,Int} = nothing
    default_discount::Union{Nothing,Float64} = nothing

    perfect_information::Union{Nothing,Bool} = nothing
    stochastic::Union{Nothing,Bool} = nothing
    simultaneous_moves::Union{Nothing,Bool} = nothing
end

game_spec(::Kernel.AbstractGame) = GameSpec()

end