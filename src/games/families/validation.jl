module GameValidation

using ..Kernel
using ..Spec
using ..Classification
using ..POSGValidation
using ..DecPOMDPValidation

export ValidationIssue
export ValidationSection
export ValidationReport

export validate_game
export validate_posg
export validate_decpomdp

export is_valid_game
export require_valid_game

export summarize_failures

# ----------------------------------------------------------------------
# Structured validation report
# ----------------------------------------------------------------------

struct ValidationIssue
    code::Symbol
    ok::Bool
    message::String
end

struct ValidationSection
    name::Symbol
    issues::Vector{ValidationIssue}
end

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

function summarize_failures(rep::ValidationReport)
    lines = String[]
    for sec in rep.sections
        for iss in sec.issues
            iss.ok || push!(lines, "[$(sec.name)] $(iss.code): $(iss.message)")
        end
    end
    return join(lines, "\n")
end

# ----------------------------------------------------------------------
# Basic kernel / metadata checks
# ----------------------------------------------------------------------

function _kernel_section(game::Kernel.AbstractGame)
    issues = ValidationIssue[
        _issue(
            :num_players_positive,
            Kernel.num_players(game) > 0,
            "Game has at least one player.",
            "Game must have at least one player."
        ),

        _issue(
            :init_state_defined,
            let ok = true
                try
                    Kernel.init_state(game)
                catch
                    ok = false
                end
                ok
            end,
            "Game provides `Kernel.init_state`.",
            "Game must implement `Kernel.init_state(game)`."
        ),
    ]

    return ValidationSection(:kernel_basics, issues)
end

function _init_state_section(game::Kernel.AbstractGame)
    state_ok = true
    state = nothing
    try
        state = Kernel.init_state(game)
    catch
        state_ok = false
    end

    if !state_ok
        issues = ValidationIssue[
            _issue(
                :init_state_available,
                false,
                "Initial state available.",
                "Could not construct initial state."
            )
        ]
        return ValidationSection(:initial_state_contract, issues)
    end

    nk_ok = true
    nk = nothing
    try
        nk = Kernel.node_kind(game, state)
    catch
        nk_ok = false
    end

    issues = ValidationIssue[
        _issue(
            :node_kind_defined,
            nk_ok,
            "Initial state has a node kind.",
            "Game must implement `Kernel.node_kind(game, state)`."
        ),
    ]

    if nk_ok && nk == Kernel.DECISION
        p_ok = true
        p = 0
        try
            p = Kernel.current_player(game, state)
        catch
            p_ok = false
        end

        push!(issues,
            _issue(
                :decision_current_player_valid,
                p_ok && (p in Kernel.player_ids(game)),
                "Decision node exposes a valid current player.",
                "Decision node must expose a valid `current_player(game, state)`."
            )
        )

    elseif nk_ok && nk == Kernel.SIMULTANEOUS
        aps_ok = true
        aps = ()
        try
            aps = Tuple(Kernel.active_players(game, state))
        catch
            aps_ok = false
        end

        strictly_ascending = true
        if aps_ok
            @inbounds for i in 2:length(aps)
                aps[i - 1] < aps[i] || (strictly_ascending = false)
            end
        end

        valid_ids = true
        if aps_ok
            ids = Kernel.player_ids(game)
            @inbounds for p in aps
                p in ids || (valid_ids = false)
            end
        end

        push!(issues,
            _issue(
                :simultaneous_active_players_defined,
                aps_ok,
                "Simultaneous node exposes active players.",
                "Simultaneous node must implement `active_players(game, state)`."
            )
        )

        push!(issues,
            _issue(
                :simultaneous_active_players_valid,
                aps_ok && valid_ids,
                "Active players are valid player ids.",
                "Active players must be valid player ids."
            )
        )

        push!(issues,
            _issue(
                :simultaneous_active_players_ascending,
                aps_ok && strictly_ascending,
                "Active players are strictly ascending.",
                "Active players must be strictly ascending."
            )
        )
    end

    return ValidationSection(:initial_state_contract, issues)
end

function _spec_section(game::Kernel.AbstractGame)
    spec_ok = true
    spec = nothing
    try
        spec = Spec.game_spec(game)
    catch
        spec_ok = false
    end

    issues = ValidationIssue[
        _issue(
            :game_spec_defined,
            spec_ok,
            "Game provides `Spec.game_spec(game)`.",
            "Game should provide `Spec.game_spec(game)`."
        )
    ]

    if spec_ok
        push!(issues,
            _issue(
                :default_discount_valid,
                isnothing(spec.default_discount) || (0.0 <= spec.default_discount <= 1.0),
                "Default discount is valid or omitted.",
                "`default_discount` must lie in [0, 1] when provided."
            )
        )

        push!(issues,
            _issue(
                :max_steps_valid,
                isnothing(spec.max_steps) || spec.max_steps > 0,
                "Max-steps metadata is valid or omitted.",
                "`max_steps` must be positive when provided."
            )
        )
    end

    return ValidationSection(:spec_metadata, issues)
end

function _classification_section(game::Kernel.AbstractGame)
    is_posg_heur = Classification.is_posg(game)
    is_decpomdp_heur = Classification.is_decpomdp(game)

    issues = ValidationIssue[
        _issue(
            :classification_consistency,
            !(is_decpomdp_heur && !is_posg_heur),
            "High-level heuristic classifications are internally consistent.",
            "A game classified as Dec-POMDP should also satisfy the POSG heuristic."
        )
    ]

    return ValidationSection(:classification_consistency, issues)
end

# ----------------------------------------------------------------------
# Generic game validation
# ----------------------------------------------------------------------

function validate_game(game::Kernel.AbstractGame)
    sections = ValidationSection[
        _kernel_section(game),
        _init_state_section(game),
        _spec_section(game),
        _classification_section(game),
    ]
    return ValidationReport(:game, _report_valid(sections), sections)
end

is_valid_game(game::Kernel.AbstractGame) = validate_game(game).valid

function require_valid_game(game::Kernel.AbstractGame)
    rep = validate_game(game)
    rep.valid && return game
    throw(ArgumentError(
        "Game failed validation.\n" * summarize_failures(rep)
    ))
end

# ----------------------------------------------------------------------
# Adapters from semantic validators
# ----------------------------------------------------------------------

function _convert_issue(iss)
    return ValidationIssue(iss.code, iss.ok, iss.message)
end

function _convert_section(sec)
    return ValidationSection(sec.name, [_convert_issue(iss) for iss in sec.issues])
end

function validate_posg(game::Kernel.AbstractGame)
    rep = POSGValidation.validate_posg(game)
    secs = [_convert_section(sec) for sec in rep.sections]
    return ValidationReport(:posg, rep.valid, secs)
end

function validate_decpomdp(game::Kernel.AbstractGame)
    rep = DecPOMDPValidation.validate_decpomdp(game)
    secs = [_convert_section(sec) for sec in rep.sections]
    return ValidationReport(:decpomdp, rep.valid, secs)
end

end