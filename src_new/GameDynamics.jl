module GameDynamics

include("core/api.jl")
include("core/traits.jl")
include("core/validation.jl")

using .CoreAPI
using .GameTraits

export CoreAPI, GameTraits

end