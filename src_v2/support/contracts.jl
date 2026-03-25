module Contracts

using ..Compiled
using ..CompiledNormalFormModels
using ..CompiledMarkovModels
using ..CompiledExtensiveModels

export ModelRole
export DIAGNOSTIC_ONLY, REFERENCE_GRADE, SOLVER_GRADE

export model_role
export is_solver_grade
export is_reference_grade
export is_diagnostic_only
export supports_analysis
export supports_exact_solvers
export supports_approx_solvers

@enum ModelRole::UInt8 begin
    DIAGNOSTIC_ONLY = 0x01
    REFERENCE_GRADE = 0x02
    SOLVER_GRADE    = 0x03
end

# ----------------------------------------------------------------------
# Default contracts
# ----------------------------------------------------------------------

model_role(::Compiled.AbstractCompiledModel) = REFERENCE_GRADE

is_solver_grade(model) = model_role(model) == SOLVER_GRADE
is_reference_grade(model) = model_role(model) == REFERENCE_GRADE
is_diagnostic_only(model) = model_role(model) == DIAGNOSTIC_ONLY

supports_analysis(::Compiled.AbstractCompiledModel) = true
supports_exact_solvers(::Compiled.AbstractCompiledModel) = false
supports_approx_solvers(::Compiled.AbstractCompiledModel) = false

# ----------------------------------------------------------------------
# Normal-form compiled models
# ----------------------------------------------------------------------

model_role(::CompiledNormalFormModels.CompiledMatrixGame) = SOLVER_GRADE
supports_exact_solvers(::CompiledNormalFormModels.CompiledMatrixGame) = true
supports_approx_solvers(::CompiledNormalFormModels.CompiledMatrixGame) = true

# ----------------------------------------------------------------------
# Markov compiled models
# ----------------------------------------------------------------------

model_role(::CompiledMarkovModels.CompiledMDP) = SOLVER_GRADE
supports_exact_solvers(::CompiledMarkovModels.CompiledMDP) = true
supports_approx_solvers(::CompiledMarkovModels.CompiledMDP) = false

model_role(::CompiledMarkovModels.CompiledZeroSumMarkovGame) = SOLVER_GRADE
supports_exact_solvers(::CompiledMarkovModels.CompiledZeroSumMarkovGame) = true
supports_approx_solvers(::CompiledMarkovModels.CompiledZeroSumMarkovGame) = false

# ----------------------------------------------------------------------
# Extensive-form compiled models
# ----------------------------------------------------------------------

function model_role(model::CompiledExtensiveModels.CompiledExtensiveGame)
    return model.has_simultaneous ? DIAGNOSTIC_ONLY : SOLVER_GRADE
end

supports_exact_solvers(::CompiledExtensiveModels.CompiledExtensiveGame) = false

function supports_approx_solvers(model::CompiledExtensiveModels.CompiledExtensiveGame)
    return !model.has_simultaneous && model.n_players == 2
end

end