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

@testset "DecisionRules" begin
    include("decision_rules/helpers_tests.jl")
    include("decision_rules/interface_tests.jl")
    include("decision_rules/internal_utils_tests.jl")
    include("decision_rules/direct_rules_tests.jl")
    include("decision_rules/lookup_rules_tests.jl")
    include("decision_rules/profiles_tests.jl")
    include("decision_rules/joint_rules_tests.jl")
end


include("learners/feedback_tests.jl")
include("learners/diagnostics_tests.jl")

include("learners/core/contexts_tests.jl")
include("learners/core/interfaces_tests.jl")

include("learners/online/bandits/exp3_tests.jl")
include("learners/online/bandits/thompson_tests.jl")
include("learners/online/bandits/ucb_tests.jl")

include("learners/online/full_information/hedge_tests.jl")
include("learners/online/full_information/ftrl_tests.jl")
include("learners/online/full_information/ftpl_tests.jl")


include("TestHarness.jl")
include("TestFixtures.jl")

include("analysis/test_encodings.jl")
include("analysis/test_reports.jl")
include("analysis/test_evaluation.jl")

include("analysis/tabular/test_traits.jl")
include("analysis/tabular/test_matrix_game.jl")
include("analysis/tabular/test_mdp.jl")
include("analysis/tabular/test_zero_sum_markov_game.jl")
include("analysis/tabular/test_extensive_tree.jl")
include("analysis/tabular/test_extensive_graph.jl")
include("analysis/tabular/test_validation.jl")

include("analysis/solvers/approx/test_common.jl")
include("analysis/solvers/approx/test_regret_matching.jl")
include("analysis/solvers/approx/test_extragradient.jl")