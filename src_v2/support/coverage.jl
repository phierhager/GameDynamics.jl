module Coverage

using ..Kernel
using ..Capabilities
using ..Classification
using ..Contracts
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

function representation_support(game::Kernel.AbstractGame)
    if Classification.is_normal_form(game)
        return STRONG
    elseif Classification.is_extensive_form(game)
        return BASELINE
    elseif Classification.is_stochastic_game(game)
        return BASELINE
    elseif Classification.is_posg(game)
        return REPRESENTATION_ONLY
    else
        return REPRESENTATION_ONLY
    end
end

function exact_solver_support(game::Kernel.AbstractGame)
    if Classification.is_normal_form(game)
        return REFERENCE
    elseif Classification.is_extensive_form(game)
        return UNSUPPORTED
    elseif Classification.is_stochastic_game(game)
        return REFERENCE
    else
        return UNSUPPORTED
    end
end

function approx_solver_support(game::Kernel.AbstractGame)
    if Classification.is_normal_form(game)
        return BASELINE
    elseif Classification.is_extensive_form(game)
        return BASELINE
    elseif Classification.is_posg(game)
        return UNSUPPORTED
    else
        return UNSUPPORTED
    end
end

compiled_support(model::CompiledNormalFormModels.CompiledMatrixGame) =
    Contracts.is_solver_grade(model) ? STRONG : REFERENCE

compiled_support(model::CompiledMarkovModels.CompiledMDP) =
    Contracts.is_solver_grade(model) ? STRONG : REFERENCE

compiled_support(model::CompiledMarkovModels.CompiledZeroSumMarkovGame) =
    Contracts.is_solver_grade(model) ? BASELINE : REFERENCE

function compiled_support(model::CompiledExtensiveModels.CompiledExtensiveGame)
    if Contracts.is_solver_grade(model)
        return STRONG
    elseif Contracts.is_diagnostic_only(model)
        return REPRESENTATION_ONLY
    else
        return REFERENCE
    end
end

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