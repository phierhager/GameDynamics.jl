module GameLab

using Random
using StaticArrays
using Statistics

# ------------------------------------------------------------------------------
# Thin bridge namespaces used by some submodules in the tree you pasted
# ------------------------------------------------------------------------------

module Exact end

module Learning
export learner_family
learner_family(::Any) = :unknown
end

# `games/families/extensive_form/behavior.jl` imports this name, but in the code
# you pasted it does not actually use anything from it.
module Interfaces end

# ------------------------------------------------------------------------------
# Core game kernel / metadata / spaces
# ------------------------------------------------------------------------------

include("games/kernel.jl")
include("games/spec.jl")
include("games/spaces.jl")
include("games/enumerative.jl")

# ------------------------------------------------------------------------------
# Decision rules
# ------------------------------------------------------------------------------

include("decision_rules/interface.jl")
include("decision_rules/internal_utils.jl")
include("decision_rules/direct_rules.jl")
include("decision_rules/lookup_rules.jl")
include("decision_rules/profiles.jl")
include("decision_rules/joint_rules.jl")

# ------------------------------------------------------------------------------
# Extensive-form helpers needed by runtime and compilation
# ------------------------------------------------------------------------------

include("games/families/extensive_form/information_states.jl")
include("games/families/extensive_form/infosets.jl")

module Infosets
import ..ExtensiveFormInfosets:
    infoset,
    uses_information_state,
    require_information_state_interface,
    infoset_kind

export infoset,
       uses_information_state,
       require_information_state_interface,
       infoset_kind
end

include("games/families/extensive_form/behavior.jl")

# ------------------------------------------------------------------------------
# Runtime
# ------------------------------------------------------------------------------

include("runtime/environment.jl")
include("runtime/trajectories.jl")
include("runtime/rule_execution.jl")
include("runtime/rollouts.jl")

# ------------------------------------------------------------------------------
# Family classification / validation
# ------------------------------------------------------------------------------

include("games/families/classification.jl")
include("games/families/stochastic/validation_common.jl")
include("games/families/stochastic/posg_validation.jl")
include("games/families/stochastic/decpomdp_validation.jl")
include("games/families/validation.jl")

# ------------------------------------------------------------------------------
# Game families
# ------------------------------------------------------------------------------

include("games/families/normal_form.jl")
include("games/families/stochastic/repeated.jl")
include("games/families/bayesian/priors.jl")

module Bayesian
import ..BayesianPriors:
    CommonPrior,
    IndependentPrior,
    player_type,
    type_profile,
    type_space,
    marginal_type_space,
    sample_type_profile,
    prior_probability,
    marginal_probability

export CommonPrior,
       IndependentPrior,
       player_type,
       type_profile,
       type_space,
       marginal_type_space,
       sample_type_profile,
       prior_probability,
       marginal_probability
end

include("games/families/bayesian/stackelberg.jl")
include("games/families/bayesian/signaling.jl")

# ------------------------------------------------------------------------------
# Learning
# ------------------------------------------------------------------------------

include("learners/feedback.jl")
include("learners/diagnostics.jl")
include("learners/core/interfaces.jl")
include("learners/core/contexts.jl")

include("learners/online/bandits/exp3.jl")
include("learners/online/bandits/thompson.jl")
include("learners/online/bandits/ucb.jl")

include("learners/online/full_information/ftpl.jl")
include("learners/online/full_information/hedge.jl")
include("learners/online/full_information/ftrl.jl")

# RL files are included, but their module names were not shown in your tree.
# Add those module objects to `_PUBLIC_MODULES` below once their names are fixed.
include("learners/rl/actor_critic.jl")
include("learners/rl/returns.jl")
include("learners/rl/replay.jl")
include("learners/rl/q_learning.jl")
include("learners/rl/self_play.jl")

# ------------------------------------------------------------------------------
# Analysis core
# ------------------------------------------------------------------------------

include("analysis/encodings.jl")
include("analysis/reports.jl")

include("analysis/tabular/traits.jl")
include("analysis/tabular/extensive_tree.jl")
include("analysis/tabular/extensive_graph.jl")
include("analysis/tabular/matrix_game.jl")
include("analysis/tabular/mdp.jl")
include("analysis/tabular/zero_sum_markov_game.jl")
include("analysis/tabular/validation.jl")

include("analysis/solvers/approx/common.jl")
include("analysis/solvers/approx/regret_matching.jl")
include("analysis/solvers/approx/extragradient.jl")
include("analysis/solvers/approx/cfr.jl")
include("analysis/solvers/approx/mccfr.jl")

include("analysis/solvers/exact/extensive_form.jl")
include("analysis/solvers/exact/normal_form.jl")
include("analysis/solvers/exact/mdp.jl")

include("analysis/tabular/compile.jl")
include("analysis/solvers/exact/markov_games.jl")

include("analysis/evaluation.jl")

# ------------------------------------------------------------------------------
# Populate the `Exact` bridge namespace
# ------------------------------------------------------------------------------

Core.eval(Exact, quote
    import ..ExactExtensiveFormSolvers: solver_family
    import ..ExactNormalFormSolvers:
        solve_zero_sum_nash,
        solve_ce,
        solve_cce,
        support_profiles,
        profile_to_linear_index,
        linear_index_to_profile
    import ..ExactMDPSolvers:
        reachable_states,
        value_iteration_mdp,
        greedy_policy_from_values
    import ..ExactMarkovGameSolvers:
        shapley_value_iteration_zero_sum

    export solver_family,
           solve_zero_sum_nash,
           solve_ce,
           solve_cce,
           support_profiles,
           profile_to_linear_index,
           linear_index_to_profile,
           reachable_states,
           value_iteration_mdp,
           greedy_policy_from_values,
           shapley_value_iteration_zero_sum
end)

# ------------------------------------------------------------------------------
# Clean top-level export flattening
#
# - exports every submodule name
# - imports/exports each submodule's public API
# - skips Base/Core names
# - skips duplicate names to avoid a polluted / ambiguous top-level namespace
# ------------------------------------------------------------------------------

const _SKIP_EXPORTS = let s = Set{Symbol}()
    union!(s, names(Base; all = true, imported = true))
    union!(s, names(Core; all = true, imported = true))
    s
end

const _SEEN_EXPORTS = Set{Symbol}()

function _export_public!(mod::Module)
    modname = nameof(mod)
    @eval export $modname

    for sym in names(mod; all = false, imported = false)
        sym === modname && continue
        sym in _SKIP_EXPORTS && continue
        sym in _SEEN_EXPORTS && continue

        @eval import .$(modname): $(sym)
        @eval export $(sym)
        push!(_SEEN_EXPORTS, sym)
    end

    return nothing
end

const _PUBLIC_MODULES = (
    Kernel,
    Spec,
    Spaces,
    Enumerative,

    DecisionRulesInterface,
    DecisionRuleInternalUtils,
    DirectDecisionRules,
    LookupDecisionRules,
    DecisionRuleProfiles,
    JointDecisionRules,

    Interfaces,
    ExtensiveFormInfosets,
    Infosets,
    ExtensiveForm,

    RuntimeEnvironment,
    RuntimeTrajectories,
    RuleExecution,
    RuntimeRollouts,

    Classification,
    StochasticGameValidationCommon,
    POSGValidation,
    DecPOMDPValidation,
    GameValidation,

    Exact,
    NormalForm,
    RepeatedGames,
    BayesianPriors,
    Bayesian,
    Stackelberg,
    Signaling,

    Learning,
    LearningFeedback,
    LearningDiagnostics,
    LearningInterfaces,
    LearningContexts,
    EXP3Learners,
    ThompsonLearners,
    UCBLearners,
    FTPLLearners,
    HedgeLearners,
    FTRLLearners,

    Encodings,
    AnalysisReports,
    TabularTraits,
    TabularExtensiveTrees,
    TabularExtensiveGraphs,
    TabularMatrixGames,
    TabularMDPs,
    TabularMarkovGames,
    TabularValidation,
    ApproxSolverCommon,
    RegretMatchingSolvers,
    ExtragradientSolvers,
    CFRSolvers,
    MCCFRSolvers,
    ExactExtensiveFormSolvers,
    ExactNormalFormSolvers,
    ExactMDPSolvers,
    TabularCompile,
    ExactMarkovGameSolvers,
    AnalysisEvaluation,
)

for mod in _PUBLIC_MODULES
    _export_public!(mod)
end

end # module GameLab