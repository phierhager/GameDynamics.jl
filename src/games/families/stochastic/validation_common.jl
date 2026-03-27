module StochasticGameValidationCommon

using ...Kernel

export ValidationIssue
export ValidationSection
export ValidationReport

export _all_ok
export _report_valid
export _issue
export _strictly_ascending
export _all_valid_player_ids
export _summarize_failures
export kernel_contract_sanity_section

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
"""
struct ValidationSection
    name::Symbol
    issues::Vector{ValidationIssue}
end

"""
Structured semantic validation report for a named model family.
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

"""
Lightweight kernel sanity check shared by POSG / Dec-POMDP validators.

This is intentionally a sanity check, not a full reachability validator.
"""
function kernel_contract_sanity_section(game::Kernel.AbstractGame)
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

end