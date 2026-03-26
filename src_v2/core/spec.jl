module Spec

using ..Kernel

export HorizonKind, EPISODIC, CONTINUING, FINITE_HORIZON
export PayoffKind, ZERO_SUM, GENERAL_SUM, CONSTANT_SUM, UNKNOWN_PAYOFF
export ObservationKind, FULL_STATE_OBSERVATION, PARTIAL_OBSERVATION, UNKNOWN_OBSERVATION
export RewardSharing, INDEPENDENT_REWARD, SHARED_REWARD, IDENTICAL_REWARD, UNKNOWN_REWARD_SHARING

export GameSpec, game_spec

@enum HorizonKind::UInt8 begin
    EPISODIC      = 0x01
    CONTINUING    = 0x02
    FINITE_HORIZON = 0x03
end

@enum PayoffKind::UInt8 begin
    ZERO_SUM       = 0x01
    GENERAL_SUM    = 0x02
    CONSTANT_SUM   = 0x03
    UNKNOWN_PAYOFF = 0x04
end

@enum ObservationKind::UInt8 begin
    FULL_STATE_OBSERVATION = 0x01
    PARTIAL_OBSERVATION    = 0x02
    UNKNOWN_OBSERVATION    = 0x03
end

@enum RewardSharing::UInt8 begin
    INDEPENDENT_REWARD     = 0x01
    SHARED_REWARD          = 0x02
    IDENTICAL_REWARD       = 0x03
    UNKNOWN_REWARD_SHARING = 0x04
end

"""
Lightweight core semantic metadata.

This remains intentionally metadata-only:
- runtime may use `max_steps` and `default_discount`
- classification/validators may use semantic fields when available
- no solver logic lives here
- no model-family-specific behavior lives here

Use `nothing` for unknown booleans rather than guessing.
"""
Base.@kwdef struct GameSpec
    horizon_kind::HorizonKind = EPISODIC
    payoff_kind::PayoffKind = UNKNOWN_PAYOFF

    max_steps::Union{Nothing,Int} = nothing
    default_discount::Union{Nothing,Float64} = nothing

    perfect_information::Union{Nothing,Bool} = nothing
    stochastic::Union{Nothing,Bool} = nothing
    simultaneous_moves::Union{Nothing,Bool} = nothing

    observation_kind::ObservationKind = UNKNOWN_OBSERVATION

    cooperative::Union{Nothing,Bool} = nothing
    reward_sharing::RewardSharing = UNKNOWN_REWARD_SHARING
end

function game_spec(::Kernel.AbstractGame)
    return GameSpec()
end

end