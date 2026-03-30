module Families

using ..Kernel

abstract type AbstractGameFamily end

struct GenericGameFamily <: AbstractGameFamily end
struct NormalFormFamily <: AbstractGameFamily end
struct SimultaneousOneShotFamily <: AbstractGameFamily end
struct ExtensiveFormFamily <: AbstractGameFamily end
struct MDPFamily <: AbstractGameFamily end
struct MarkovGameFamily <: AbstractGameFamily end

game_family(::Type{<:Kernel.AbstractGame}) = GenericGameFamily()
game_family(game::Kernel.AbstractGame) = game_family(typeof(game))

end