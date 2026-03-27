module DecPOMDPValidation

using ...Kernel
using ...Spec
using .POSGValidation
using ..StochasticGameValidationCommon

export ValidationIssue
export ValidationSection
export ValidationReport

export validate_decpomdp
export is_valid_decpomdp
export require_valid_decpomdp

const ValidationIssue = StochasticGameValidationCommon.ValidationIssue
const ValidationSection = StochasticGameValidationCommon.ValidationSection
const ValidationReport = StochasticGameValidationCommon.ValidationReport

const _all_ok = StochasticGameValidationCommon._all_ok
const _report_valid = StochasticGameValidationCommon._report_valid
const _issue = StochasticGameValidationCommon._issue
const _summarize_failures = StochasticGameValidationCommon._summarize_failures
const kernel_contract_sanity_section = StochasticGameValidationCommon.kernel_contract_sanity_section

# ----------------------------------------------------------------------
# Dec-POMDP validation sections
# ----------------------------------------------------------------------

function game_semantics_section(game::Kernel.AbstractGame)
    spec = Spec.game_spec(game)
    posg_report = POSGValidation.validate_posg(game)

    posg_semantics_ok =
        first(sec for sec in posg_report.sections if sec.name == :game_semantics) |>
        sec -> _all_ok(sec.issues)

    shared_reward_ok =
        spec.reward_sharing == Spec.SHARED_REWARD ||
        spec.reward_sharing == Spec.IDENTICAL_REWARD

    issues = ValidationIssue[
        _issue(
            :base_posg_semantics,
            posg_semantics_ok,
            "Game satisfies base POSG semantic checks.",
            "DecPOMDP semantics require the game to satisfy the base POSG semantic checks."
        ),

        _issue(
            :cooperative,
            spec.cooperative === true,
            "Game is explicitly cooperative.",
            "DecPOMDP semantics expect `Spec.game_spec(game).cooperative === true`."
        ),

        _issue(
            :shared_reward_structure,
            shared_reward_ok,
            "Reward-sharing metadata is compatible with cooperative DecPOMDP semantics.",
            "DecPOMDP semantics expect shared/identical reward metadata."
        ),

        _issue(
            :multiagent,
            Kernel.num_players(game) >= 2,
            "Game has at least two players.",
            "DecPOMDP semantics require at least two players."
        ),
    ]

    return ValidationSection(:game_semantics, issues)
end

function policy_conventions_section(game::Kernel.AbstractGame)
    issues = ValidationIssue[
        _issue(
            :canonical_observation_memoryless_profile,
            true,
            "Game is compatible with canonical observation-based memoryless DecPOMDP profiles.",
            "Game is compatible with canonical observation-based memoryless DecPOMDP profiles."
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
            "Richer history-dependent policy classes can be layered separately from the canonical DecPOMDP profile.",
            "Richer history-dependent policy classes can be layered separately from the canonical DecPOMDP profile."
        ),
    ]

    return ValidationSection(:policy_profile_conventions, issues)
end

"""
Validate whether a game satisfies the library's semantic Dec-POMDP contract.

This is a report-first API. Dec-POMDP validity is treated as:
- base POSG semantic validity
- cooperative/shared-reward structure
- separate policy/profile convention compatibility reporting
"""
function validate_decpomdp(game::Kernel.AbstractGame)
    sections = ValidationSection[
        game_semantics_section(game),
        policy_conventions_section(game),
        kernel_contract_sanity_section(game),
    ]
    return ValidationReport(:decpomdp, _report_valid(sections), sections)
end

is_valid_decpomdp(game::Kernel.AbstractGame) = validate_decpomdp(game).valid

function require_valid_decpomdp(game::Kernel.AbstractGame)
    rep = validate_decpomdp(game)
    rep.valid && return game
    throw(ArgumentError(
        "Game failed DecPOMDP validation.\n" * _summarize_failures(rep)
    ))
end

end