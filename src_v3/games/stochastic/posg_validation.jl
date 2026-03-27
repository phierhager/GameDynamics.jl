module POSGValidation

using ...Kernel
using ...Spec
using ..StochasticGameValidationCommon

export ValidationIssue
export ValidationSection
export ValidationReport

export validate_posg
export is_valid_posg
export require_valid_posg

const ValidationIssue = StochasticGameValidationCommon.ValidationIssue
const ValidationSection = StochasticGameValidationCommon.ValidationSection
const ValidationReport = StochasticGameValidationCommon.ValidationReport

const _all_ok = StochasticGameValidationCommon._all_ok
const _report_valid = StochasticGameValidationCommon._report_valid
const _issue = StochasticGameValidationCommon._issue
const _summarize_failures = StochasticGameValidationCommon._summarize_failures
const kernel_contract_sanity_section = StochasticGameValidationCommon.kernel_contract_sanity_section

# ----------------------------------------------------------------------
# POSG validation sections
# ----------------------------------------------------------------------

function game_semantics_section(game::Kernel.AbstractGame)
    spec = Spec.game_spec(game)

    issues = ValidationIssue[
        _issue(
            :stochastic,
            spec.stochastic === true,
            "Game is explicitly stochastic.",
            "POSG semantics expect `Spec.game_spec(game).stochastic === true`."
        ),

        _issue(
            :imperfect_information,
            spec.perfect_information === false,
            "Game is explicitly imperfect-information.",
            "POSG semantics expect `Spec.game_spec(game).perfect_information === false`."
        ),

        _issue(
            :simultaneous_semantics,
            spec.simultaneous_moves === true,
            "Game declares simultaneous-move semantics.",
            "POSG semantics expect `Spec.game_spec(game).simultaneous_moves === true`."
        ),

        _issue(
            :observation_kind,
            spec.observation_kind == Spec.PARTIAL_OBSERVATION ||
            spec.observation_kind == Spec.UNKNOWN_OBSERVATION,
            spec.observation_kind == Spec.PARTIAL_OBSERVATION ?
                "Game is explicitly marked as partially observed." :
                "Observation metadata is unknown, which is weakly compatible with POSG semantics.",
            "POSG semantics are inconsistent with `Spec.FULL_STATE_OBSERVATION` metadata."
        ),

        _issue(
            :multiagent,
            Kernel.num_players(game) >= 2,
            "Game has at least two players.",
            "POSG semantics require at least two players."
        ),
    ]

    return ValidationSection(:game_semantics, issues)
end

function policy_conventions_section(game::Kernel.AbstractGame)
    spec = Spec.game_spec(game)

    issues = ValidationIssue[
        _issue(
            :canonical_observation_memoryless_profile,
            spec.observation_kind != Spec.FULL_STATE_OBSERVATION,
            "Game is compatible with canonical observation-based memoryless POSG profiles.",
            "Canonical observation-based memoryless POSG profiles are awkward when metadata says full-state observation."
        ),

        _issue(
            :state_policy_profile,
            true,
            "`StatePolicyProfile` remains a valid runtime/control abstraction.",
            "`StatePolicyProfile` remains a valid runtime/control abstraction."
        ),

        _issue(
            :history_dependent_extensions,
            true,
            "Richer history-dependent policy classes can be layered separately from the canonical POSG profile.",
            "Richer history-dependent policy classes can be layered separately from the canonical POSG profile."
        ),

        _issue(
            :partial_activity_supported,
            true,
            "Partial active-player sets at simultaneous nodes are compatible with the kernel simultaneous-action model.",
            "Partial active-player sets at simultaneous nodes are compatible with the kernel simultaneous-action model."
        ),
    ]

    return ValidationSection(:policy_profile_conventions, issues)
end

"""
Validate whether a game satisfies the library's semantic POSG contract.

This is a report-first API:
- semantic validity is checked from `Spec` and core game metadata
- policy/profile compatibility is reported separately
- lightweight kernel sanity checks are reported separately
"""
function validate_posg(game::Kernel.AbstractGame)
    sections = ValidationSection[
        game_semantics_section(game),
        policy_conventions_section(game),
        kernel_contract_sanity_section(game),
    ]
    return ValidationReport(:posg, _report_valid(sections), sections)
end

is_valid_posg(game::Kernel.AbstractGame) = validate_posg(game).valid

function require_valid_posg(game::Kernel.AbstractGame)
    rep = validate_posg(game)
    rep.valid && return game
    throw(ArgumentError(
        "Game failed POSG validation.\n" * _summarize_failures(rep)
    ))
end

end