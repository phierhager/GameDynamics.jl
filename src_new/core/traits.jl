module GameTraits

import ..CoreAPI: AbstractGame # Assuming this is loaded from your api.jl

# Export trait types
export AbstractInformationTrait, PerfectInformation, ImperfectInformation
export AbstractDynamicsTrait, Sequential, Simultaneous
export AbstractChanceTrait, Deterministic, Stochastic
export AbstractUtilityTrait, ZeroSum, ConstantSum, GeneralSum, IdenticalUtility
export AbstractHorizonTrait, FiniteHorizon, InfiniteHorizon

# Export trait functions
export information_trait, dynamics_trait, chance_trait, utility_trait, horizon_trait
export num_players, utility_bounds

# ==========================================
# 1. Information Visibility
# ==========================================
abstract type AbstractInformationTrait end
struct PerfectInformation <: AbstractInformationTrait end
struct ImperfectInformation <: AbstractInformationTrait end

"""
    information_trait(::Type{<:AbstractGame})

Does the game have hidden information (e.g., Poker) or is everything public (e.g., Chess)?
Default is `ImperfectInformation` as it is the safer, more general assumption.
"""
information_trait(::Type{<:AbstractGame}) = ImperfectInformation()

# ==========================================
# 2. Turn Dynamics
# ==========================================
abstract type AbstractDynamicsTrait end
struct Sequential <: AbstractDynamicsTrait end
struct Simultaneous <: AbstractDynamicsTrait end

"""
    dynamics_trait(::Type{<:AbstractGame})

Do players move one after another, or at the exact same time (e.g., Rock-Paper-Scissors)?
"""
dynamics_trait(::Type{<:AbstractGame}) = Sequential()

# ==========================================
# 3. Chance & Randomness
# ==========================================
abstract type AbstractChanceTrait end
struct Deterministic <: AbstractChanceTrait end
struct Stochastic <: AbstractChanceTrait end

"""
    chance_trait(::Type{<:AbstractGame})

Does the game involve dice rolls, shuffled decks, or a stochastic environment?
"""
chance_trait(::Type{<:AbstractGame}) = Deterministic()

# ==========================================
# 4. Utility / Payoff Structure
# ==========================================
abstract type AbstractUtilityTrait end
struct ZeroSum <: AbstractUtilityTrait end
struct ConstantSum <: AbstractUtilityTrait end
struct IdenticalUtility <: AbstractUtilityTrait end # Fully Cooperative
struct GeneralSum <: AbstractUtilityTrait end

"""
    utility_trait(::Type{<:AbstractGame})

How do player rewards relate to one another? Algorithms like Minimax strictly require ZeroSum.
"""
utility_trait(::Type{<:AbstractGame}) = GeneralSum()

# ==========================================
# 5. Horizon / Termination
# ==========================================
abstract type AbstractHorizonTrait end
struct FiniteHorizon <: AbstractHorizonTrait end
struct InfiniteHorizon <: AbstractHorizonTrait end

"""
    horizon_trait(::Type{<:AbstractGame})

Is the game guaranteed to end in a finite number of steps?
"""
horizon_trait(::Type{<:AbstractGame}) = FiniteHorizon()

# ==========================================
# 6. Quantitative Metadata (Hard Requirements)
# ==========================================

"""
    num_players(game::AbstractGame) -> Int

Returns the number of acting players in the game (excluding the Chance/Environment player).
"""
function num_players end 

"""
    utility_bounds(game::AbstractGame) -> Tuple{Float64, Float64}

Returns `(min_utility, max_utility)`. 
Crucial for algorithms that need to normalize returns (like CFR+ or AlphaZero).
"""
function utility_bounds end

end # module