module TestHarness

module LearningDiagnostics
export LearnerTrace
export cumulative_regret
export average_reward
export average_utility
export empirical_action_frequencies!

Base.@kwdef struct LearnerTrace
    t::Int
    cumulative_utility::Float64
    cumulative_reward::Float64
    best_fixed_utility::Float64
end

cumulative_regret(tr::LearnerTrace) = tr.best_fixed_utility - tr.cumulative_utility
average_reward(tr::LearnerTrace) = tr.t == 0 ? 0.0 : tr.cumulative_reward / tr.t
average_utility(tr::LearnerTrace) = tr.t == 0 ? 0.0 : tr.cumulative_utility / tr.t

function empirical_action_frequencies!(dest::AbstractVector{Float64}, counts::AbstractVector{<:Integer})
    length(dest) == length(counts) || throw(ArgumentError("Destination length mismatch."))

    z = sum(counts)
    if z > 0
        @inbounds for i in eachindex(dest, counts)
            dest[i] = counts[i] / z
        end
    else
        v = 1 / length(dest)
        fill!(dest, v)
    end
    return dest
end

end

include("../src/analysis/encodings.jl")
include("../src/analysis/reports.jl")

include("../src/analysis/tabular/traits.jl")
include("../src/analysis/tabular/matrix_game.jl")
include("../src/analysis/tabular/mdp.jl")
include("../src/analysis/tabular/zero_sum_markov_game.jl")
include("../src/analysis/tabular/extensive_tree.jl")
include("../src/analysis/tabular/extensive_graph.jl")

include("../src/analysis/solvers/approx/common.jl")
include("../src/analysis/solvers/approx/regret_matching.jl")
include("../src/analysis/solvers/approx/extragradient.jl")

include("../src/analysis/evaluation.jl")
include("../src/analysis/tabular/validation.jl")

end
