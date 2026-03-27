using Test
using GameLab

include("helpers/mock_games.jl")

include("games/kernel_tests.jl")
include("games/spaces_tests.jl")
include("games/spec_tests.jl")
include("games/enumerative_tests.jl")
include("games/families/classification_tests.jl")
include("games/families/validation_tests.jl")
include("games/families/normal_form_tests.jl")
include("games/families/repeated_tests.jl")
include("games/families/stackelberg_tests.jl")
include("games/families/priors_tests.jl")
include("games/families/signaling_tests.jl")
include("games/families/extensive_form_tests.jl")