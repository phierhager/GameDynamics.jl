module Spec

using ..Kernel

export HorizonKind, EPISODIC, CONTINUING
export PlayerModelKind, FIXED_PLAYERS, POPULATION_PLAYERS

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

"""
Descriptive metadata only.

These fields do not alter the kernel stepping contract.
`max_steps` may be consulted by runtime helpers such as `Runtime.default_episode_limit`,
but kernel `step` itself does not implement truncation.
"""
Base.@kwdef struct GameSpec
    perfect_information::Bool = true
    perfect_recall::Bool = true
    stochastic::Bool = false
    simultaneous_moves::Bool = false
    zero_sum::Bool = false
    general_sum::Bool = false

    horizon_kind::HorizonKind = EPISODIC
    player_model::PlayerModelKind = FIXED_PLAYERS

    max_steps::Union{Nothing, Int} = nothing
    default_discount::Union{Nothing, Float64} = nothing

    mechanism::MechanismSpec = MechanismSpec()
    teams::Vector{TeamSpec} = TeamSpec[]
    coalitions::Vector{CoalitionSpec} = CoalitionSpec[]
end

game_spec(::Kernel.AbstractGame) = GameSpec()

end