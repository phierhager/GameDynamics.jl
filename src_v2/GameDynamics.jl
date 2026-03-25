module GameDynamics

include("core/kernel.jl")
include("core/spaces.jl")
include("core/capabilities.jl")
include("core/exact.jl")
include("core/spec.jl")
include("core/runtime.jl")
include("core/views.jl")

include("theory/strategies.jl")
include("theory/bayesian.jl")
include("theory/normalform.jl")
include("theory/classification.jl")
include("theory/extensive_form.jl")
include("theory/repeated.jl")
include("theory/stochastic_games.jl")
include("theory/signaling.jl")
include("theory/stackelberg.jl")
include("theory/posg.jl")

include("domains/domains.jl")
include("domains/auctions.jl")
include("domains/congestion.jl")
include("domains/graphical.jl")
include("domains/repeated_auctions.jl")

include("encodings/encodings.jl")

include("compiled/compiled.jl")
include("compiled/markov_models.jl")
include("compiled/normalform_models.jl")
include("compiled/extensive_models.jl")
include("compiled/policies.jl")

include("support/contracts.jl")
include("support/coverage.jl")

include("learning/learning.jl")
include("learning/interfaces.jl")
include("learning/feedback.jl")
include("learning/diagnostics.jl")
include("learning/full_information/hedge.jl")
include("learning/full_information/ftrl.jl")
include("learning/full_information/ftpl.jl")
include("learning/bandits/exp3.jl")
include("learning/bandits/ucb.jl")
include("learning/bandits/thompson.jl")

include("solvers/exact/exact.jl")
include("solvers/exact/normalform.jl")
include("solvers/exact/extensive_form.jl")
include("solvers/exact/markov.jl")

include("solvers/approx/approx.jl")
include("solvers/approx/common.jl")
include("solvers/approx/regret_matching.jl")
include("solvers/approx/first_order_matrix_games.jl")
include("solvers/approx/cfr.jl")
include("solvers/approx/mccfr.jl")
include("solvers/approx/matrix_diagnostics.jl")

include("analysis/analysis.jl")
include("analysis/normalform.jl")
include("analysis/extensive_form.jl")

export Kernel, Spaces, Capabilities, Exact, Spec, Runtime, Views

export Strategies, Bayesian, NormalForm
export Classification, ExtensiveForm, RepeatedGames, StochasticGames
export Signaling, Stackelberg, POSG

export Domains, Auctions, Congestion, Graphical, RepeatedAuctions

export Encodings

export Compiled, CompiledMarkovModels, CompiledNormalFormModels, CompiledExtensiveModels
export Contracts, Coverage
export CompiledPolicies

export Learning, LearningInterfaces, LearningFeedback, LearningDiagnostics
export HedgeLearners, FTRLLearners, FTPLLearners
export EXP3Learners, UCBLearners, ThompsonLearners

export ExactSolvers, ExactNormalFormSolvers, ExactExtensiveFormSolvers, ExactMarkovSolvers
export ApproxSolvers, ApproxSolverCommon
export RegretMatchingSolvers, FirstOrderMatrixGameSolvers
export CFRSolvers, MCCFRSolvers
export MatrixApproxDiagnostics

export Analysis, NormalFormAnalysis, ExtensiveFormAnalysis

end