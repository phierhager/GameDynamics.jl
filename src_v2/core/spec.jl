module Spec

using ..Kernel

export HorizonKind, EPISODIC, CONTINUING
export PlayerModelKind, FIXED_PLAYERS, POPULATION_PLAYERS
export PayoffKind, ZERO_SUM, GENERAL_SUM, CONSTANT_SUM, UNKNOWN_PAYOFF
export TeamSpec, CoalitionSpec, MechanismSpec, GameSpec
export game_spec

@enum HorizonKind::UInt8 begin
    EPISODIC = 0x01
    CONTINUING = 0x02
end

@enum PlayerModelKind::UInt8 begin
    FIXED_PLAYERS = 0x01
    POPULATION_PLAYERS = 0x02
end

@enum PayoffKind::UInt8 begin
    ZERO_SUM = 0x01
    GENERAL_SUM = 0x02
    CONSTANT_SUM = 0x03
    UNKNOWN_PAYOFF = 0x04
end

struct TeamSpec
    id::Symbol
    members::Vector{Int}
end

struct CoalitionSpec
    id::Symbol
    members::Vector{Int}
    transferable_utility::Bool
end

Base.@kwdef struct MechanismSpec
    public_recommendations::Bool = false
    private_recommendations::Bool = false
    cheap_talk::Bool = false
    transfers::Bool = false
end

Base.@kwdef struct GameSpec
    perfect_information::Union{Nothing,Bool} = nothing
    perfect_recall::Union{Nothing,Bool} = nothing
    stochastic::Union{Nothing,Bool} = nothing
    simultaneous_moves::Union{Nothing,Bool} = nothing

    payoff_kind::PayoffKind = UNKNOWN_PAYOFF
    horizon_kind::HorizonKind = EPISODIC
    player_model::PlayerModelKind = FIXED_PLAYERS

    max_steps::Union{Nothing,Int} = nothing
    default_discount::Union{Nothing,Float64} = nothing

    mechanism::MechanismSpec = MechanismSpec()
    teams::Vector{TeamSpec} = TeamSpec[]
    coalitions::Vector{CoalitionSpec} = CoalitionSpec[]
end

game_spec(::Kernel.AbstractGame) = GameSpec()

end