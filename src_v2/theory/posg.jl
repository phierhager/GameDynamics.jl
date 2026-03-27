module POSG

using ..Kernel
using ..Spec

export ValidationIssue
export ValidationSection
export ValidationReport

export validate_posg
export validate_decpomdp

export is_valid_posg
export is_valid_decpomdp

export require_valid_posg
export require_valid_decpomdp

# ----------------------------------------------------------------------
# Structured validation report
# ----------------------------------------------------------------------

"""
One validation result inside a section.

Fields:
- `code`: stable symbolic identifier for the check
- `ok`: whether the check passed
- `message`: human-readable explanation
"""
struct ValidationIssue
    code::Symbol
    ok::Bool
    message::String
end

"""
A named group of related validation issues.

Typical section names:
- `:game_semantics`
- `:policy_profile_conventions`
- `:kernel_contract_sanity`
"""
struct ValidationSection
    name::Symbol
    issues::Vector{ValidationIssue}
end

"""
Structured semantic validation report for a named model family.

Fields:
- `family`: semantic family, e.g. `:posg` or `:decpomdp`
- `valid`: overall pass/fail
- `sections`: grouped issue sections
"""
struct ValidationReport
    family::Symbol
    valid::Bool
    sections::Vector{ValidationSection}
end

@inline _all_ok(xs::Vector{ValidationIssue}) = all(x -> x.ok, xs)
@inline _report_valid(sections::Vector{ValidationSection}) = all(sec -> _all_ok(sec.issues), sections)

Base.show(io::IO, rep::ValidationReport) =
    print(io, "ValidationReport(family=", rep.family,
              ", valid=", rep.valid,
              ", sections=", length(rep.sections), ")")

# ----------------------------------------------------------------------
# Internal formatting helpers
# ----------------------------------------------------------------------

@inline function _issue(code::Symbol, ok::Bool, ok_msg::AbstractString, bad_msg::AbstractString)
    return ValidationIssue(code, ok, String(ok ? ok_msg : bad_msg))
end

@inline function _strictly_ascending(xs)
    @inbounds for i in 2:length(xs)
        xs[i - 1] < xs[i] || return false
    end
    return true
end

@inline function _all_valid_player_ids(game::Kernel.AbstractGame, aps)
    ids = Kernel.player_ids(game)
    @inbounds for p in aps
        p in ids || return false
    end
    return true
end

function _summarize_failures(rep::ValidationReport)
    lines = String[]
    for sec in rep.sections
        for iss in sec.issues
            iss.ok || push!(lines, "[$(sec.name)] $(iss.code): $(iss.message)")
        end
    end
    return join(lines, "\n")
end

# ----------------------------------------------------------------------
# POSG validation
# ----------------------------------------------------------------------

function _game_semantics_posg(game::Kernel.AbstractGame)
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

function _policy_conventions_posg(game::Kernel.AbstractGame)
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

function _kernel_contract_sanity(game::Kernel.AbstractGame)
    # This is intentionally a sanity check, not a full reachability validator.
    state = Kernel.init_state(game)
    nk = Kernel.node_kind(game, state)

    if nk == Kernel.SIMULTANEOUS
        aps = Tuple(Kernel.active_players(game, state))
        issues = ValidationIssue[
            _issue(
                :active_players_valid_ids,
                _all_valid_player_ids(game, aps),
                "Initial simultaneous node uses valid player ids.",
                "Initial simultaneous node contains invalid active player ids."
            ),

            _issue(
                :active_players_ascending,
                _strictly_ascending(aps),
                "Initial simultaneous node uses strictly ascending active-player order.",
                "Initial simultaneous node violates the required strictly ascending active-player order."
            ),
        ]
    else
        issues = ValidationIssue[
            _issue(
                :initial_state_not_simultaneous,
                true,
                "Initial state is not simultaneous; no initial active-player ordering sanity check was needed.",
                "Initial state is not simultaneous; no initial active-player ordering sanity check was needed."
            ),
        ]
    end

    return ValidationSection(:kernel_contract_sanity, issues)
end

"""
Validate whether a game satisfies the library's semantic POSG contract.

This is a report-first API:
- semantic validity is checked from `Spec` and core game metadata
- policy/profile compatibility is reported separately
- lightweight kernel sanity checks are reported separately

This validator does not require a dedicated POSG wrapper type.
"""
function validate_posg(game::Kernel.AbstractGame)
    sections = ValidationSection[
        _game_semantics_posg(game),
        _policy_conventions_posg(game),
        _kernel_contract_sanity(game),
    ]
    return ValidationReport(:posg, _report_valid(sections), sections)
end

# ----------------------------------------------------------------------
# Dec-POMDP validation
# ----------------------------------------------------------------------

function _game_semantics_decpomdp(game::Kernel.AbstractGame)
    spec = Spec.game_spec(game)
    posg_report = validate_posg(game)

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

function _policy_conventions_decpomdp(game::Kernel.AbstractGame)
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
        _game_semantics_decpomdp(game),
        _policy_conventions_decpomdp(game),
        _kernel_contract_sanity(game),
    ]
    return ValidationReport(:decpomdp, _report_valid(sections), sections)
end

# ----------------------------------------------------------------------
# Convenience wrappers
# ----------------------------------------------------------------------

is_valid_posg(game::Kernel.AbstractGame) = validate_posg(game).valid
is_valid_decpomdp(game::Kernel.AbstractGame) = validate_decpomdp(game).valid

function require_valid_posg(game::Kernel.AbstractGame)
    rep = validate_posg(game)
    rep.valid && return game
    throw(ArgumentError(
        "Game failed POSG validation.\n" * _summarize_failures(rep)
    ))
end

function require_valid_decpomdp(game::Kernel.AbstractGame)
    rep = validate_decpomdp(game)
    rep.valid && return game
    throw(ArgumentError(
        "Game failed DecPOMDP validation.\n" * _summarize_failures(rep)
    ))
end

end