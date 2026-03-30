module GameLab

# ------------------------------------------------------------------------------
# Core
# ------------------------------------------------------------------------------

include("common/records.jl")

include("games/kernel.jl")
include("games/enumerative.jl")
include("games/families.jl")
include("games/spaces.jl")

# ------------------------------------------------------------------------------
# Strategies
# ------------------------------------------------------------------------------

include("strategies/internal/utils.jl")
include("strategies/interface.jl")
include("strategies/local.jl")
include("strategies/joint.jl")
include("strategies/record.jl")
include("strategies/indexed.jl")
include("strategies/profiles.jl")

# ------------------------------------------------------------------------------
# Games
# ------------------------------------------------------------------------------

include("games/normal_form/basic.jl")
include("games/normal_form/matrix.jl")

# ------------------------------------------------------------------------------
# Learners
# ------------------------------------------------------------------------------

include("learners/interfaces.jl")
include("learners/diagnostics.jl")

include("learners/online/bandits/exp3.jl")
include("learners/online/bandits/thompson.jl")
include("learners/online/bandits/ucb.jl")

include("learners/online/full_information/ftpl.jl")
include("learners/online/full_information/hedge.jl")
include("learners/online/full_information/ftrl.jl")

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

include("utils/encodings.jl")
include("utils/evaluation.jl")

# ------------------------------------------------------------------------------
# Tabular
# ------------------------------------------------------------------------------

include("tabular/traits.jl")
include("tabular/matrix_game.jl")
include("tabular/mdp.jl")
include("tabular/zero_sum_markov_game.jl")
include("tabular/extensive_tree.jl")
include("tabular/extensive_graph.jl")
include("tabular/compile.jl")
include("tabular/validation.jl")

# ------------------------------------------------------------------------------
# Runtime
# ------------------------------------------------------------------------------

include("runtime/execution.jl")
include("runtime/rollouts.jl")

# ------------------------------------------------------------------------------
# Solvers
# ------------------------------------------------------------------------------

include("solvers/api.jl")

include("solvers/approx/common.jl")
include("solvers/approx/regret_matching.jl")
include("solvers/approx/extragradient.jl")
include("solvers/approx/cfr.jl")
include("solvers/approx/mccfr.jl")

include("solvers/exact/extensive_form.jl")
include("solvers/exact/normal_form.jl")
include("solvers/exact/mdp.jl")
include("solvers/exact/markov_games.jl")

include("solvers/prepare.jl")
include("solvers/solve.jl")

# ------------------------------------------------------------------------------
# Top-level module exports
# ------------------------------------------------------------------------------

export RuntimeRecords
export Kernel
export Enumerative
export Families
export Spaces

export StrategyInternalUtils
export StrategyInterface
export LocalStrategies
export JointStrategies
export RecordStrategies
export IndexedStrategies
export StrategyProfiles

export NormalForm
export MatrixGames

export LearningInterfaces
export LearningDiagnostics
export EXP3Learners
export ThompsonLearners
export UCBLearners
export FTPLLearners
export HedgeLearners
export FTRLLearners

export Encodings
export AnalysisEvaluation

export TabularTraits
export TabularMatrixGames
export TabularMDPs
export TabularMarkovGames
export TabularExtensiveTrees
export TabularExtensiveGraphs
export TabularCompile
export TabularValidation

export RuntimeStrategyExecution
export RuntimeRollouts

export SolverAPI
export SolverPrepare
export SolveDispatch

export ApproxSolverCommon
export RegretMatchingSolvers
export ExtragradientSolvers
export CFRSolvers
export MCCFRSolvers

export ExactExtensiveFormSolvers
export ExactNormalFormSolvers
export ExactMDPSolvers
export ExactMarkovGameSolvers

# ------------------------------------------------------------------------------
# User-facing alias
# ------------------------------------------------------------------------------

export solve
const solve = SolverAPI.solve

end # module GameLab