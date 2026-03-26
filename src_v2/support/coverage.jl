module Coverage

using ..Kernel
using ..Capabilities
using ..Contracts
using ..Compiled
using ..CompiledExtensiveModels
using ..CompiledMarkovModels
using ..CompiledNormalFormModels

export SupportLevel, UNSUPPORTED, REPRESENTATION_ONLY, REFERENCE, BASELINE, STRONG
export representation_support
export exact_solver_support
export approx_solver_support
export compiled_support
export support_report

@enum SupportLevel::UInt8 begin
    UNSUPPORTED         = 0x00
    REPRESENTATION_ONLY = 0x01
    REFERENCE           = 0x02
    BASELINE            = 0x03
    STRONG              = 0x04
end

@inline function _compiled_support_from_contracts(model::Compiled.AbstractCompiledModel)
    if Contracts.is_diagnostic_only(model)
        return REPRESENTATION_ONLY
    elseif Contracts.is_solver_grade(model)
        if Contracts.supports_exact_solvers(model) && Contracts.supports_approx_solvers(model)
            return STRONG
        elseif Contracts.supports_exact_solvers(model) || Contracts.supports_approx_solvers(model)
            return BASELINE
        else
            return REFERENCE
        end
    else
        return REFERENCE
    end
end

function representation_support(game::Kernel.AbstractGame)
    if Contracts.supports_exact_solvers(game) || Contracts.supports_approx_solvers(game)
        return BASELINE
    elseif Contracts.supports_analysis(game)
        return REPRESENTATION_ONLY
    else
        return UNSUPPORTED
    end
end

exact_solver_support(game::Kernel.AbstractGame) =
    Contracts.supports_exact_solvers(game) ? REFERENCE : UNSUPPORTED

approx_solver_support(game::Kernel.AbstractGame) =
    Contracts.supports_approx_solvers(game) ? BASELINE : UNSUPPORTED

compiled_support(model::Compiled.AbstractCompiledModel) =
    _compiled_support_from_contracts(model)

function support_report(game::Kernel.AbstractGame)
    return (
        representation = representation_support(game),
        exact = exact_solver_support(game),
        approximate = approx_solver_support(game),
    )
end

function support_report(model::CompiledExtensiveModels.CompiledExtensiveGame)
    return (
        compiled = compiled_support(model),
        role = Contracts.model_role(model),
        supports_analysis = Contracts.supports_analysis(model),
        supports_approx = Contracts.supports_approx_solvers(model),
        supports_exact = Contracts.supports_exact_solvers(model),
        is_tree = model.is_tree,
        simultaneous = model.has_simultaneous,
    )
end

function support_report(model::CompiledNormalFormModels.CompiledMatrixGame)
    return (
        compiled = compiled_support(model),
        role = Contracts.model_role(model),
        supports_analysis = Contracts.supports_analysis(model),
        supports_approx = Contracts.supports_approx_solvers(model),
        supports_exact = Contracts.supports_exact_solvers(model),
    )
end

function support_report(model::CompiledMarkovModels.CompiledMDP)
    return (
        compiled = compiled_support(model),
        role = Contracts.model_role(model),
        supports_analysis = Contracts.supports_analysis(model),
        supports_approx = Contracts.supports_approx_solvers(model),
        supports_exact = Contracts.supports_exact_solvers(model),
    )
end

function support_report(model::CompiledMarkovModels.CompiledZeroSumMarkovGame)
    return (
        compiled = compiled_support(model),
        role = Contracts.model_role(model),
        supports_analysis = Contracts.supports_analysis(model),
        supports_approx = Contracts.supports_approx_solvers(model),
        supports_exact = Contracts.supports_exact_solvers(model),
    )
end

end