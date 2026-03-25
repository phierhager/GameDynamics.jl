module CompiledExtensiveModels

using ..Compiled
using ..Kernel
using ..Exact
using ..ExtensiveForm
using ..Encodings

export CompiledExtensiveGame
export CompiledExtensiveBuilder
export compile_extensive_game
export compile_extensive_game_typed
export n_nodes, n_infosets, node_action_count
export NODE_TERMINAL, NODE_CHANCE, NODE_DECISION, NODE_SIMULTANEOUS
export infoset_player, node_infoset, node_player, active_players
export chance_probabilities, action_range, infoset_range
export infoset_action_labels, has_simultaneous_nodes

const NODE_TERMINAL     = UInt8(0x00)
const NODE_CHANCE       = UInt8(0x01)
const NODE_DECISION     = UInt8(0x02)
const NODE_SIMULTANEOUS = UInt8(0x03)

struct CompiledExtensiveGame{NE,IE,AL} <: Compiled.AbstractCompiledExtensiveFormModel
    n_players::Int
    has_simultaneous::Bool

    n_nodes::Int
    node_kind::Vector{UInt8}
    node_player::Vector{Int}
    node_infoset::Vector{Int}
    node_first::Vector{Int}
    node_len::Vector{Int}
    child::Vector{Int}

    slot_label::Vector{AL}
    action_id_within_infoset::Vector{Int}
    chance_prob::Vector{Float64}

    n_infosets::Int
    infoset_player::Vector{Int}
    infoset_num_actions::Vector{Int}
    infoset_offset::Vector{Int}
    infoset_action_label::Vector{AL}

    node_active_first::Vector{Int}
    node_active_len::Vector{Int}
    active_player_ids::Vector{Int}

    n_terminals::Int
    reward_first::Vector{Int}
    terminal_payoffs::Vector{Float64}

    node_encoder::NE
    infoset_encoder::IE
    root_node::Int
end

mutable struct CompiledExtensiveBuilder{S,LT}
    states::Vector{S}
    queue::Vector{Int}

    node_kind::Vector{UInt8}
    node_player::Vector{Int}
    node_infoset::Vector{Int}
    node_first::Vector{Int}
    node_len::Vector{Int}
    reward_first::Vector{Int}

    child::Vector{Int}
    slot_label::Vector{LT}
    action_id_within_infoset::Vector{Int}
    chance_prob::Vector{Float64}

    infoset_player_vec::Vector{Int}
    infoset_num_actions::Vector{Int}
    infoset_labels_tmp::Vector{Vector{LT}}

    node_active_first::Vector{Int}
    node_active_len::Vector{Int}
    active_player_ids::Vector{Int}

    terminal_payoff_vec::Vector{Float64}
end

function CompiledExtensiveBuilder(::Type{S}, ::Type{LT};
                                  state_hint::Int = 4096,
                                  edge_hint::Int = 4096) where {S,LT}
    states = S[]
    sizehint!(states, state_hint)
    queue = Int[]
    sizehint!(queue, state_hint)

    node_kind = UInt8[]; sizehint!(node_kind, state_hint)
    node_player = Int[]; sizehint!(node_player, state_hint)
    node_infoset = Int[]; sizehint!(node_infoset, state_hint)
    node_first = Int[]; sizehint!(node_first, state_hint)
    node_len = Int[]; sizehint!(node_len, state_hint)
    reward_first = Int[]; sizehint!(reward_first, state_hint)

    child = Int[]; sizehint!(child, edge_hint)
    slot_label = LT[]; sizehint!(slot_label, edge_hint)
    action_id_within_infoset = Int[]; sizehint!(action_id_within_infoset, edge_hint)
    chance_prob = Float64[]; sizehint!(chance_prob, edge_hint)

    infoset_player_vec = Int[]
    infoset_num_actions = Int[]
    infoset_labels_tmp = Vector{Vector{LT}}()

    node_active_first = Int[]; sizehint!(node_active_first, state_hint)
    node_active_len = Int[]; sizehint!(node_active_len, state_hint)
    active_player_ids = Int[]; sizehint!(active_player_ids, edge_hint)

    terminal_payoff_vec = Float64[]; sizehint!(terminal_payoff_vec, state_hint)

    return CompiledExtensiveBuilder{S,LT}(
        states, queue,
        node_kind, node_player, node_infoset, node_first, node_len, reward_first,
        child, slot_label, action_id_within_infoset, chance_prob,
        infoset_player_vec, infoset_num_actions, infoset_labels_tmp,
        node_active_first, node_active_len, active_player_ids,
        terminal_payoff_vec,
    )
end

n_nodes(g::CompiledExtensiveGame) = g.n_nodes
n_infosets(g::CompiledExtensiveGame) = g.n_infosets
node_action_count(g::CompiledExtensiveGame, node::Int) = g.node_len[node]
infoset_player(g::CompiledExtensiveGame, infoset::Int) = g.infoset_player[infoset]
node_infoset(g::CompiledExtensiveGame, node::Int) = g.node_infoset[node]
node_player(g::CompiledExtensiveGame, node::Int) = g.node_player[node]
has_simultaneous_nodes(g::CompiledExtensiveGame) = g.has_simultaneous

@inline function action_range(g::CompiledExtensiveGame, node::Int)
    a0 = g.node_first[node]
    return a0:(a0 + g.node_len[node] - 1)
end

@inline function infoset_range(g::CompiledExtensiveGame, infoset::Int)
    i0 = g.infoset_offset[infoset]
    return i0:(g.infoset_offset[infoset + 1] - 1)
end

chance_probabilities(g::CompiledExtensiveGame, node::Int) =
    @view g.chance_prob[action_range(g, node)]

function active_players(g::CompiledExtensiveGame, node::Int)
    k0 = g.node_active_first[node]
    len = g.node_active_len[node]
    len == 0 && return ()
    return Tuple(g.active_player_ids[k0:(k0 + len - 1)])
end

infoset_action_labels(g::CompiledExtensiveGame, infoset::Int) =
    g.infoset_action_label[infoset_range(g, infoset)]

@inline _reward_component(r, p::Int) = r isa Real ? (p == 1 ? Float64(r) : 0.0) : Float64(r[p])

@inline function _build_node_encoder(T, root_state)
    enc = Encodings.DenseEncoder{T}()
    Encodings.encode!(enc, root_state)
    return enc
end

function _build_infoset_offsets(infoset_num_actions::Vector{Int})
    I = length(infoset_num_actions)
    offs = Vector{Int}(undef, I + 1)
    offs[1] = 1
    @inbounds for i in 1:I
        offs[i + 1] = offs[i] + infoset_num_actions[i]
    end
    return offs
end

@inline _materialize(xs::AbstractVector) = xs
@inline _materialize(xs::Tuple) = xs
@inline _materialize(xs::Base.OneTo) = xs
_materialize(xs) = collect(xs)

@inline _length_fast(xs::AbstractVector) = length(xs)
@inline _length_fast(xs::Tuple) = length(xs)
@inline _length_fast(xs::Base.OneTo) = length(xs)
_length_fast(xs) = length(collect(xs))

@inline function _get_or_add_state!(builder, node_enc, ns, max_nodes::Int)
    if Encodings.has_encoding(node_enc, ns)
        return Encodings.encode(node_enc, ns)
    end
    nid = Encodings.encode!(node_enc, ns)
    length(builder.states) < max_nodes || error("Node budget exceeded in compile_extensive_game.")
    push!(builder.states, ns)
    push!(builder.queue, nid)
    return nid
end

@inline function _push_terminal!(builder, state, N::Int)
    push!(builder.node_kind, NODE_TERMINAL)
    push!(builder.node_player, 0)
    push!(builder.node_infoset, 0)
    push!(builder.node_first, 1)
    push!(builder.node_len, 0)
    push!(builder.node_active_first, 1)
    push!(builder.node_active_len, 0)

    n_terminals = length(builder.reward_first) + 1
    push!(builder.reward_first, (n_terminals - 1) * N + 1)

    rewards = Exact.terminal_payoffs(state[1], state[2])
    @inbounds for p in 1:N
        push!(builder.terminal_payoff_vec, _reward_component(rewards, p))
    end
    return nothing
end

function _finalize_compiled(builder,
                            node_enc,
                            infoset_enc,
                            N::Int,
                            saw_simultaneous::Bool,
                            label_type::Type)
    offs = _build_infoset_offsets(builder.infoset_num_actions)
    total_infoset_actions = offs[end] - 1
    infoset_action_label = Vector{label_type}(undef, total_infoset_actions)

    @inbounds for infoset in 1:length(builder.infoset_num_actions)
        i0 = offs[infoset]
        labs = builder.infoset_labels_tmp[infoset]
        for a in 1:length(labs)
            infoset_action_label[i0 + a - 1] = labs[a]
        end
    end

    return CompiledExtensiveGame{typeof(node_enc),typeof(infoset_enc),label_type}(
        N,
        saw_simultaneous,
        length(builder.states),
        builder.node_kind,
        builder.node_player,
        builder.node_infoset,
        builder.node_first,
        builder.node_len,
        builder.child,
        builder.slot_label,
        builder.action_id_within_infoset,
        builder.chance_prob,
        length(builder.infoset_player_vec),
        builder.infoset_player_vec,
        builder.infoset_num_actions,
        offs,
        infoset_action_label,
        builder.node_active_first,
        builder.node_active_len,
        builder.active_player_ids,
        length(builder.reward_first),
        builder.reward_first,
        builder.terminal_payoff_vec,
        node_enc,
        infoset_enc,
        1,
    )
end

"""
Typed fast path.

Use this whenever infosets and action labels can be given concrete types.
This avoids the `Any` default path and improves compiler inference for the
compiled extensive-form model.

By default, simultaneous nodes are rejected unless `expand_simultaneous=true`.
That keeps the compiled tree solver-grade by default.
"""
function compile_extensive_game_typed(game::Kernel.AbstractGame,
                                      root_state,
                                      ::Type{IT},
                                      ::Type{LT};
                                      max_nodes::Int = 1_000_000,
                                      state_hint::Int = 4096,
                                      edge_hint::Int = 4096,
                                      expand_simultaneous::Bool = false) where {IT,LT}
    N = Kernel.num_players(game)
    S = typeof(root_state)

    builder = CompiledExtensiveBuilder(S, LT;
                                       state_hint = min(state_hint, max_nodes),
                                       edge_hint = edge_hint)

    node_enc = _build_node_encoder(S, root_state)
    infoset_enc = Encodings.DenseEncoder{IT}()
    Encodings.sizehint!(infoset_enc, state_hint)

    push!(builder.states, root_state)
    push!(builder.queue, 1)

    saw_simultaneous = false

    head = 1
    while head <= length(builder.queue)
        node_id = builder.queue[head]
        head += 1

        state = builder.states[node_id]
        nk = Kernel.node_kind(game, state)

        if nk == Kernel.TERMINAL
            push!(builder.node_kind, NODE_TERMINAL)
            push!(builder.node_player, 0)
            push!(builder.node_infoset, 0)
            push!(builder.node_first, 1)
            push!(builder.node_len, 0)
            push!(builder.node_active_first, 1)
            push!(builder.node_active_len, 0)

            n_terminals = length(builder.reward_first) + 1
            push!(builder.reward_first, (n_terminals - 1) * N + 1)

            rewards = Exact.terminal_payoffs(game, state)
            @inbounds for p in 1:N
                push!(builder.terminal_payoff_vec, _reward_component(rewards, p))
            end

        elseif nk == Kernel.CHANCE
            outcomes = _materialize(Exact.chance_outcomes(game, state))
            a0 = length(builder.child) + 1

            @inbounds for k in eachindex(outcomes)
                event, pr = outcomes[k]
                ns, _, _ = Kernel.step(game, state, Kernel.ChanceOutcome(event))
                nsid = _get_or_add_state!(builder, node_enc, ns, max_nodes)
                push!(builder.child, nsid)
                push!(builder.slot_label, convert(LT, event))
                push!(builder.action_id_within_infoset, 0)
                push!(builder.chance_prob, Float64(pr))
            end

            push!(builder.node_kind, NODE_CHANCE)
            push!(builder.node_player, 0)
            push!(builder.node_infoset, 0)
            push!(builder.node_first, a0)
            push!(builder.node_len, length(outcomes))
            push!(builder.node_active_first, 1)
            push!(builder.node_active_len, 0)
            push!(builder.reward_first, 0)

        elseif nk == Kernel.DECISION
            p = Kernel.current_player(game, state)
            info = convert(IT, ExtensiveForm.infoset(game, state, p))

            infoset_id = if Encodings.has_encoding(infoset_enc, info)
                Encodings.encode(infoset_enc, info)
            else
                iid = Encodings.encode!(infoset_enc, info)
                push!(builder.infoset_player_vec, p)
                push!(builder.infoset_num_actions, 0)
                push!(builder.infoset_labels_tmp, LT[])
                iid
            end

            acts = _materialize(Kernel.legal_actions(game, state, p))
            nA = _length_fast(acts)
            a0 = length(builder.child) + 1

            if builder.infoset_num_actions[infoset_id] == 0
                builder.infoset_num_actions[infoset_id] = nA
                labs = Vector{LT}(undef, nA)
                @inbounds for i in 1:nA
                    labs[i] = convert(LT, acts[i])
                end
                builder.infoset_labels_tmp[infoset_id] = labs
            elseif builder.infoset_num_actions[infoset_id] != nA
                throw(ArgumentError("Infoset action count mismatch for infoset $infoset_id."))
            end

            @inbounds for local_aid in 1:nA
                a = acts[local_aid]
                ns, _, _ = Kernel.step(game, state, a)
                nsid = _get_or_add_state!(builder, node_enc, ns, max_nodes)
                push!(builder.child, nsid)
                push!(builder.slot_label, convert(LT, a))
                push!(builder.action_id_within_infoset, local_aid)
                push!(builder.chance_prob, 0.0)
            end

            push!(builder.node_kind, NODE_DECISION)
            push!(builder.node_player, p)
            push!(builder.node_infoset, infoset_id)
            push!(builder.node_first, a0)
            push!(builder.node_len, nA)
            push!(builder.node_active_first, 1)
            push!(builder.node_active_len, 0)
            push!(builder.reward_first, 0)

        elseif nk == Kernel.SIMULTANEOUS
            saw_simultaneous = true

            expand_simultaneous ||
                throw(ArgumentError("compile_extensive_game_typed encountered a simultaneous node. By default these are rejected because compiled tree solvers currently support decision/chance/terminal trees only. Pass `expand_simultaneous=true` for representation/diagnostic use."))

            aps = _materialize(Kernel.active_players(game, state))
            af0 = length(builder.active_player_ids) + 1
            append!(builder.active_player_ids, aps)

            action_lists = map(p -> _materialize(Kernel.legal_actions(game, state, p)), aps)
            a0 = length(builder.child) + 1

            for joint in Iterators.product(Tuple(action_lists)...)
                jt = Tuple(joint)
                ja = Kernel.JointAction(jt)
                ns, _, _ = Kernel.step(game, state, ja)
                nsid = _get_or_add_state!(builder, node_enc, ns, max_nodes)
                push!(builder.child, nsid)
                push!(builder.slot_label, convert(LT, jt))
                push!(builder.action_id_within_infoset, 0)
                push!(builder.chance_prob, 0.0)
            end

            push!(builder.node_kind, NODE_SIMULTANEOUS)
            push!(builder.node_player, 0)
            push!(builder.node_infoset, 0)
            push!(builder.node_first, a0)
            push!(builder.node_len, length(builder.child) - a0 + 1)
            push!(builder.node_active_first, af0)
            push!(builder.node_active_len, length(aps))
            push!(builder.reward_first, 0)

        else
            throw(ArgumentError("Unsupported node kind in compile_extensive_game_typed."))
        end
    end

    return _finalize_compiled(builder, node_enc, infoset_enc, N, saw_simultaneous, LT)
end

"""
Generic convenience path.

This preserves the old API shape, but is intentionally less performance-oriented.
For performance-sensitive usage, prefer `compile_extensive_game_typed`.

Defaults:
- `expand_simultaneous = false`
- `infoset_type = Any`
- `label_type = Any`
"""
function compile_extensive_game(game::Kernel.AbstractGame,
                                root_state;
                                max_nodes::Int = 1_000_000,
                                infoset_type::Type = Any,
                                label_type::Type = Any,
                                state_hint::Int = 4096,
                                edge_hint::Int = 4096,
                                expand_simultaneous::Bool = false)
    return compile_extensive_game_typed(game, root_state, infoset_type, label_type;
                                        max_nodes = max_nodes,
                                        state_hint = state_hint,
                                        edge_hint = edge_hint,
                                        expand_simultaneous = expand_simultaneous)
end

end