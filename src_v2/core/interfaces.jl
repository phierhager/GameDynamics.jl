module Interfaces

using ..Kernel
using ..Exact

export AbstractInterfaceMarker
export InformationStateInterface
export PublicObservationInterface
export PublicStateInterface
export ChanceOutcomesInterface
export TransitionKernelInterface
export ObservationKernelInterface
export TerminalPayoffsInterface

export supported_interfaces
export supports_interface

export ValidationIssue
export ValidationReport

export validate_interface
export validate_optional_interfaces

export require_interface
export ensure_interface

# ----------------------------------------------------------------------
# Typed optional-interface markers
# ----------------------------------------------------------------------

abstract type AbstractInterfaceMarker end

struct InformationStateInterface   <: AbstractInterfaceMarker end
struct PublicObservationInterface  <: AbstractInterfaceMarker end
struct PublicStateInterface        <: AbstractInterfaceMarker end
struct ChanceOutcomesInterface     <: AbstractInterfaceMarker end
struct TransitionKernelInterface   <: AbstractInterfaceMarker end
struct ObservationKernelInterface  <: AbstractInterfaceMarker end
struct TerminalPayoffsInterface    <: AbstractInterfaceMarker end

"""
Declare the optional interfaces supported by a game type.

Example:

    Interfaces.supported_interfaces(::Type{MyGame}) = (
        Interfaces.InformationStateInterface,
        Interfaces.PublicObservationInterface,
    )

This is a declaration of intent, not proof.
"""
supported_interfaces(::Type{<:Kernel.AbstractGame}) = ()
supported_interfaces(game::Kernel.AbstractGame) = supported_interfaces(typeof(game))
@inline supports_interface(::Type{G}, ::Type{T}) where {G<:Kernel.AbstractGame,T<:AbstractInterfaceMarker} =
    T in supported_interfaces(G)

@inline supports_interface(game::Kernel.AbstractGame, ::Type{T}) where {T<:AbstractInterfaceMarker} =
    supports_interface(typeof(game), T)

# ----------------------------------------------------------------------
# Structured validation
# ----------------------------------------------------------------------

struct ValidationIssue
    interface::Type{<:AbstractInterfaceMarker}
    ok::Bool
    message::String
end

struct ValidationReport
    valid::Bool
    issues::Vector{ValidationIssue}
end

@inline _ok(::Type{T}, msg::AbstractString) where {T<:AbstractInterfaceMarker} =
    ValidationIssue(T, true, String(msg))

@inline _bad(::Type{T}, msg::AbstractString) where {T<:AbstractInterfaceMarker} =
    ValidationIssue(T, false, String(msg))

Base.show(io::IO, rep::ValidationReport) =
    print(io, "ValidationReport(valid=", rep.valid, ", issues=", length(rep.issues), ")")

# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

@inline function _first_player(game::Kernel.AbstractGame)
    ids = Tuple(Kernel.player_ids(game))
    isempty(ids) && throw(ArgumentError("Game has no players."))
    return first(ids)
end

_representative_state(game::Kernel.AbstractGame) = Kernel.init_state(game)

"""
Build a representative `(state, action)` pair from the initial state.
Returns `(ok::Bool, payload)` where payload is either `(state, action)` or an error message.
"""
function _representative_transition(game::Kernel.AbstractGame)
    state = _representative_state(game)
    nk = Kernel.node_kind(game, state)

    if nk == Kernel.TERMINAL
        return false, "Cannot build a representative transition from an immediately terminal initial state."
    elseif nk == Kernel.CHANCE
        return true, (state, Kernel.SampleChance())
    elseif nk == Kernel.DECISION
        p = Kernel.current_player(game, state)
        legal = Kernel.legal_actions(game, state, p)
        isempty(legal) && return false, "Decision node has empty legal action set on the representative state."
        return true, (state, first(legal))
    elseif nk == Kernel.SIMULTANEOUS
        aps = Tuple(Kernel.active_players(game, state))
        acts = try
            ntuple(i -> begin
                p = aps[i]
                legal = Kernel.legal_actions(game, state, p)
                isempty(legal) && throw(ArgumentError(
                    "Simultaneous node has empty legal action set for active player $p."
                ))
                first(legal)
            end, length(aps))
        catch err
            return false, sprint(showerror, err)
        end
        return true, (state, Kernel.joint_action(acts))
    else
        return false, "Representative state has unknown node kind $nk."
    end
end

@inline function _declared_or_bad(game::Kernel.AbstractGame, ::Type{T}) where {T<:AbstractInterfaceMarker}
    supports_interface(game, T) || return _bad(T, "Game does not declare $(nameof(T)) support.")
    return nothing
end

function _call_check(::Type{T}, label::AbstractString, f, args...; success_msg::AbstractString = "") where {T<:AbstractInterfaceMarker}
    applicable(f, args...) || return _bad(
        T,
        "Game declares $(nameof(T)) but does not implement $label."
    )

    try
        f(args...)
        return _ok(T, isempty(success_msg) ? "$(nameof(T)) is declared and $label is callable." : String(success_msg))
    catch err
        return _bad(
            T,
            "Game declares $(nameof(T)) but $label failed: $(sprint(showerror, err))."
        )
    end
end

# ----------------------------------------------------------------------
# Per-interface validation
# ----------------------------------------------------------------------

function validate_interface(game::Kernel.AbstractGame, ::Type{InformationStateInterface})
    declared = _declared_or_bad(game, InformationStateInterface)
    isnothing(declared) || return declared

    state = _representative_state(game)
    p = _first_player(game)

    return _call_check(
        InformationStateInterface,
        "Exact.information_state(game, state, player)",
        Exact.information_state, game, state, p;
        success_msg = "InformationStateInterface is declared and Exact.information_state is callable on a representative state."
    )
end

function validate_interface(game::Kernel.AbstractGame, ::Type{PublicObservationInterface})
    declared = _declared_or_bad(game, PublicObservationInterface)
    isnothing(declared) || return declared

    state = _representative_state(game)

    return _call_check(
        PublicObservationInterface,
        "Exact.public_observation(game, state)",
        Exact.public_observation, game, state;
        success_msg = "PublicObservationInterface is declared and Exact.public_observation is callable on a representative state."
    )
end

function validate_interface(game::Kernel.AbstractGame, ::Type{PublicStateInterface})
    declared = _declared_or_bad(game, PublicStateInterface)
    isnothing(declared) || return declared

    state = _representative_state(game)

    return _call_check(
        PublicStateInterface,
        "Exact.public_state(game, state)",
        Exact.public_state, game, state;
        success_msg = "PublicStateInterface is declared and Exact.public_state is callable on a representative state."
    )
end

function validate_interface(game::Kernel.AbstractGame, ::Type{ChanceOutcomesInterface})
    declared = _declared_or_bad(game, ChanceOutcomesInterface)
    isnothing(declared) || return declared

    state = _representative_state(game)

    return _call_check(
        ChanceOutcomesInterface,
        "Exact.chance_outcomes(game, state)",
        Exact.chance_outcomes, game, state;
        success_msg = "ChanceOutcomesInterface is declared and Exact.chance_outcomes is callable on a representative state."
    )
end

function validate_interface(game::Kernel.AbstractGame, ::Type{TransitionKernelInterface})
    declared = _declared_or_bad(game, TransitionKernelInterface)
    isnothing(declared) || return declared

    ok, payload = _representative_transition(game)
    ok || return _bad(TransitionKernelInterface, payload)

    state, action = payload

    return _call_check(
        TransitionKernelInterface,
        "Exact.transition_kernel(game, state, action)",
        Exact.transition_kernel, game, state, action;
        success_msg = "TransitionKernelInterface is declared and Exact.transition_kernel is callable on a representative transition."
    )
end

function validate_interface(game::Kernel.AbstractGame, ::Type{ObservationKernelInterface})
    declared = _declared_or_bad(game, ObservationKernelInterface)
    isnothing(declared) || return declared

    ok, payload = _representative_transition(game)
    ok || return _bad(ObservationKernelInterface, payload)

    state, action = payload

    next_state = try
        ns, _ = Kernel.step(game, state, action)
        ns
    catch err
        return _bad(
            ObservationKernelInterface,
            "Game declares ObservationKernelInterface but could not compute a representative next state via Kernel.step: $(sprint(showerror, err))."
        )
    end

    return _call_check(
        ObservationKernelInterface,
        "Exact.observation_kernel(game, state, action, next_state)",
        Exact.observation_kernel, game, state, action, next_state;
        success_msg = "ObservationKernelInterface is declared and Exact.observation_kernel is callable on a representative transition."
    )
end

function validate_interface(game::Kernel.AbstractGame, ::Type{TerminalPayoffsInterface})
    declared = _declared_or_bad(game, TerminalPayoffsInterface)
    isnothing(declared) || return declared

    state = _representative_state(game)

    if !Kernel.is_terminal(game, state)
        return _ok(
            TerminalPayoffsInterface,
            "TerminalPayoffsInterface is declared. Full validation requires a representative terminal state."
        )
    end

    return _call_check(
        TerminalPayoffsInterface,
        "Exact.terminal_payoffs(game, state)",
        Exact.terminal_payoffs, game, state;
        success_msg = "TerminalPayoffsInterface is declared and Exact.terminal_payoffs is callable on a representative terminal state."
    )
end

# ----------------------------------------------------------------------
# Aggregate validation
# ----------------------------------------------------------------------

function validate_optional_interfaces(game::Kernel.AbstractGame)
    markers = supported_interfaces(game)
    issues = ValidationIssue[validate_interface(game, m) for m in markers]
    return ValidationReport(all(x -> x.ok, issues), issues)
end

# ----------------------------------------------------------------------
# Require / ensure
# ----------------------------------------------------------------------

function require_interface(game::Kernel.AbstractGame, marker::Type{T}) where {T<:AbstractInterfaceMarker}
    issue = validate_interface(game, marker)
    issue.ok && return game
    throw(ArgumentError(issue.message))
end

ensure_interface(game::Kernel.AbstractGame, marker::Type{T}) where {T<:AbstractInterfaceMarker} =
    require_interface(game, marker)

end