module TabularCompile

using ..Kernel
using ..NormalForm
using ..ExtensiveFormInfosets
using ..Encodings

using ..TabularMatrixGames
using ..TabularMDPs
using ..TabularMarkovGames
using ..TabularExtensiveTrees
using ..TabularExtensiveGraphs

export compile_matrix_game
export compile_mdp
export compile_zero_sum_markov_game
export compile_extensive_tree
export compile_extensive_tree_typed
export compile_extensive_graph
export compile_extensive_graph_typed

@inline _scalar_reward(r) = r isa Real ? Float64(r) : Float64(r[1])
@inline _reward_component(r, p::Int) = r isa Real ? (p == 1 ? Float64(r) : 0.0) : Float64(r[p])

@inline _action_count(xs::Base.OneTo) = length(xs)
@inline _action_count(xs::AbstractVector) = length(xs)
@inline _action_count(xs::Tuple) = length(xs)
@inline _action_count(xs) = length(collect(xs))

@inline _materialize(xs::AbstractVector) = xs
@inline _materialize(xs::Tuple) = xs
@inline _materialize(xs::Base.OneTo) = xs
_materialize(xs) = collect(xs)

@inline _length_fast(xs::AbstractVector) = length(xs)
@inline _length_fast(xs::Tuple) = length(xs)
@inline _length_fast(xs::Base.OneTo) = length(xs)
_length_fast(xs) = length(collect(xs))

function _build_state_encoder(states::AbstractVector{S}) where {S}
    enc = Encodings.DenseEncoder{S}()
    Encodings.sizehint!(enc, length(states))
    @inbounds for s in states
        Encodings.encode!(enc, s)
    end
    return enc
end

function compile_matrix_game(game::NormalForm.NormalFormGame{2})
    U1 = Matrix{Float64}(game.payoffs[1])
    U2 = Matrix{Float64}(game.payoffs[2])
    m, n = size(U1)
    return TabularMatrixGames.TabularMatrixGame(U1, U2, m, n)
end

function compile_mdp(game::Kernel.AbstractGame, states)
    Kernel.num_players(game) == 1 ||
        throw(ArgumentError("compile_mdp requires a single-player game."))

    Kernel.action_mode(typeof(game)) === Kernel.IndexedActions ||
        throw(ArgumentError("compile_mdp requires IndexedActions."))

    state_vec = collect(states)
    S = eltype(state_vec)
    enc = _build_state_encoder(state_vec)
    nS = length(state_vec)

    n_actions_vec = Vector{Int}(undef, nS)
    action_ptr = Vector{Int}(undef, nS + 1)

    total_sa = 0
    @inbounds for sidx in 1:nS
        s = state_vec[sidx]
        nk = Kernel.node_kind(game, s)

        nA = if nk == Kernel.TERMINAL
            0
        elseif nk == Kernel.CHANCE
            throw(ArgumentError("compile_mdp does not support explicit chance states. Fold stochasticity into Enumerative.transition_kernel for decision actions."))
        elseif nk == Kernel.SIMULTANEOUS
            throw(ArgumentError("compile_mdp does not support simultaneous states. Use a Markov-game compiler instead."))
        else
            _action_count(Kernel.legal_actions(game, s, 1))
        end

        n_actions_vec[sidx] = nA
        action_ptr[sidx] = total_sa + 1
        total_sa += nA
    end
    action_ptr[end] = total_sa + 1

    trans_ptr = Vector{Int}(undef, total_sa + 1)
    next_state = Int[]
    prob = Float64[]
    reward = Float64[]
    action_label = Int[]

    sizehint!(next_state, max(total_sa, 16))
    sizehint!(prob, max(total_sa, 16))
    sizehint!(reward, max(total_sa, 16))
    sizehint!(action_label, max(total_sa, 16))

    sa_idx = 1
    @inbounds for sidx in 1:nS
        s = state_vec[sidx]
        nk = Kernel.node_kind(game, s)
        nk == Kernel.TERMINAL && continue

        for a in Kernel.legal_actions(game, s, 1)
            trans_ptr[sa_idx] = length(next_state) + 1
            push!(action_label, a)
            for entry in Enumerative.transition_kernel(game, s, a)
                ns = entry[1]
                push!(next_state, Encodings.encode(enc, ns))
                push!(prob, Float64(entry[2]))
                push!(reward, _scalar_reward(entry[3]))
            end
            sa_idx += 1
        end
    end
    trans_ptr[end] = length(next_state) + 1

    return TabularMDPs.TabularMDP{typeof(enc),S}(
        nS,
        n_actions_vec,
        action_ptr,
        trans_ptr,
        next_state,
        prob,
        reward,
        action_label,
        enc,
        state_vec,
    )
end

function compile_zero_sum_markov_game(game::Kernel.AbstractGame, states)
    Kernel.num_players(game) == 2 ||
        throw(ArgumentError("compile_zero_sum_markov_game requires a 2-player game."))

    Kernel.action_mode(typeof(game)) === Kernel.IndexedActions ||
        throw(ArgumentError("compile_zero_sum_markov_game requires IndexedActions."))

    state_vec = collect(states)
    S = eltype(state_vec)
    enc = _build_state_encoder(state_vec)
    nS = length(state_vec)

    a1s = zeros(Int, nS)
    a2s = zeros(Int, nS)
    pair_ptr = Vector{Int}(undef, nS + 1)
    legal_actions_p1 = Vector{Vector{Int}}(undef, nS)
    legal_actions_p2 = Vector{Vector{Int}}(undef, nS)

    total_pairs = 0
    @inbounds for sidx in 1:nS
        s = state_vec[sidx]
        nk = Kernel.node_kind(game, s)

        if nk == Kernel.SIMULTANEOUS
            A1 = collect(Kernel.legal_actions(game, s, 1))
            A2 = collect(Kernel.legal_actions(game, s, 2))

            legal_actions_p1[sidx] = A1
            legal_actions_p2[sidx] = A2

            a1 = length(A1)
            a2 = length(A2)

            a1s[sidx] = a1
            a2s[sidx] = a2
            pair_ptr[sidx] = total_pairs + 1
            total_pairs += a1 * a2
        elseif nk == Kernel.TERMINAL
            legal_actions_p1[sidx] = Int[]
            legal_actions_p2[sidx] = Int[]
            pair_ptr[sidx] = total_pairs + 1
        else
            throw(ArgumentError("compile_zero_sum_markov_game only supports simultaneous and terminal states."))
        end
    end
    pair_ptr[end] = total_pairs + 1

    trans_ptr = Vector{Int}(undef, total_pairs + 1)
    next_state = Int[]
    prob = Float64[]
    reward = Float64[]

    sizehint!(next_state, max(total_pairs, 16))
    sizehint!(prob, max(total_pairs, 16))
    sizehint!(reward, max(total_pairs, 16))

    pair_idx = 1
    @inbounds for sidx in 1:nS
        s = state_vec[sidx]
        Kernel.node_kind(game, s) == Kernel.SIMULTANEOUS || continue

        A1 = legal_actions_p1[sidx]
        A2 = legal_actions_p2[sidx]

        for i in eachindex(A1), j in eachindex(A2)
            trans_ptr[pair_idx] = length(next_state) + 1
            ja = Kernel.JointAction((A1[i], A2[j]))
            for entry in Enumerative.transition_kernel(game, s, ja)
                ns = entry[1]
                push!(next_state, Encodings.encode(enc, ns))
                push!(prob, Float64(entry[2]))
                push!(reward, _scalar_reward(entry[3]))
            end
            pair_idx += 1
        end
    end
    trans_ptr[end] = length(next_state) + 1

    return TabularMarkovGames.TabularZeroSumMarkovGame{typeof(enc),S,Vector{Int},Vector{Int}}(
        nS,
        a1s,
        a2s,
        pair_ptr,
        trans_ptr,
        next_state,
        prob,
        reward,
        enc,
        state_vec,
        legal_actions_p1,
        legal_actions_p2,
    )
end

mutable struct _CompiledExtensiveBuilder{S,LT}
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

function _CompiledExtensiveBuilder(::Type{S}, ::Type{LT};
                                   state_hint::Int = 4096,
                                   edge_hint::Int = 4096) where {S,LT}
    states = S[]; sizehint!(states, state_hint)
    queue = Int[]; sizehint!(queue, state_hint)

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

    return _CompiledExtensiveBuilder{S,LT}(
        states, queue,
        node_kind, node_player, node_infoset, node_first, node_len, reward_first,
        child, slot_label, action_id_within_infoset, chance_prob,
        infoset_player_vec, infoset_num_actions, infoset_labels_tmp,
        node_active_first, node_active_len, active_player_ids,
        terminal_payoff_vec,
    )
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

@inline function _build_node_encoder(::Type{T}, root_state) where {T}
    enc = Encodings.DenseEncoder{T}()
    Encodings.encode!(enc, root_state)
    return enc
end

@inline function _add_tree_state!(builder, ns, max_nodes::Int)
    length(builder.states) < max_nodes || error("Node budget exceeded in compile_extensive_tree_typed.")
    nid = length(builder.states) + 1
    push!(builder.states, ns)
    push!(builder.queue, nid)
    return nid
end

@inline function _get_or_add_graph_state!(builder, node_enc, ns, max_nodes::Int)
    if Encodings.has_encoding(node_enc, ns)
        return Encodings.encode(node_enc, ns)
    end
    nid = Encodings.encode!(node_enc, ns)
    length(builder.states) < max_nodes || error("Node budget exceeded in compile_extensive_graph_typed.")
    push!(builder.states, ns)
    push!(builder.queue, nid)
    return nid
end

function _ensure_infoset_action_labels!(builder, infoset_id::Int, acts, ::Type{LT}) where {LT}
    nA = length(acts)

    if builder.infoset_num_actions[infoset_id] == 0
        builder.infoset_num_actions[infoset_id] = nA
        labs = Vector{LT}(undef, nA)
        @inbounds for i in 1:nA
            labs[i] = convert(LT, acts[i])
        end
        builder.infoset_labels_tmp[infoset_id] = labs
    else
        builder.infoset_num_actions[infoset_id] == nA ||
            throw(ArgumentError("Infoset action count mismatch for infoset $infoset_id."))

        labs = builder.infoset_labels_tmp[infoset_id]
        @inbounds for i in 1:nA
            ai = convert(LT, acts[i])
            labs[i] == ai || throw(ArgumentError(
                "Infoset action label/order mismatch for infoset $infoset_id at local action $i."
            ))
        end
    end

    return nothing
end

function _finalize_tree(builder, infoset_enc, N::Int, saw_simultaneous::Bool, ::Type{LT}) where {LT}
    offs = _build_infoset_offsets(builder.infoset_num_actions)
    total_infoset_actions = offs[end] - 1
    infoset_action_label = Vector{LT}(undef, total_infoset_actions)

    @inbounds for infoset in 1:length(builder.infoset_num_actions)
        i0 = offs[infoset]
        labs = builder.infoset_labels_tmp[infoset]
        for a in 1:length(labs)
            infoset_action_label[i0 + a - 1] = labs[a]
        end
    end

    return TabularExtensiveTrees.TabularExtensiveTree{typeof(infoset_enc),LT}(
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
        infoset_enc,
        1,
    )
end

function _finalize_graph(builder, node_enc, infoset_enc, N::Int, saw_simultaneous::Bool, ::Type{LT}) where {LT}
    offs = _build_infoset_offsets(builder.infoset_num_actions)
    total_infoset_actions = offs[end] - 1
    infoset_action_label = Vector{LT}(undef, total_infoset_actions)

    @inbounds for infoset in 1:length(builder.infoset_num_actions)
        i0 = offs[infoset]
        labs = builder.infoset_labels_tmp[infoset]
        for a in 1:length(labs)
            infoset_action_label[i0 + a - 1] = labs[a]
        end
    end

    return TabularExtensiveGraphs.TabularExtensiveGraph{typeof(node_enc),typeof(infoset_enc),LT}(
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

function compile_extensive_tree_typed(game::Kernel.AbstractGame,
                                      root_state,
                                      ::Type{IT},
                                      ::Type{LT};
                                      max_nodes::Int = 1_000_000,
                                      state_hint::Int = 4096,
                                      edge_hint::Int = 4096,
                                      expand_simultaneous::Bool = false) where {IT,LT}
    N = Kernel.num_players(game)
    S = typeof(root_state)

    builder = _CompiledExtensiveBuilder(S, LT;
                                        state_hint = min(state_hint, max_nodes),
                                        edge_hint = edge_hint)

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
            push!(builder.node_kind, TabularExtensiveTrees.NODE_TERMINAL)
            push!(builder.node_player, 0)
            push!(builder.node_infoset, 0)
            push!(builder.node_first, 1)
            push!(builder.node_len, 0)
            push!(builder.node_active_first, 1)
            push!(builder.node_active_len, 0)

            push!(builder.reward_first, length(builder.terminal_payoff_vec) + 1)
            rewards = Enumerative.terminal_payoffs(game, state)
            @inbounds for p in 1:N
                push!(builder.terminal_payoff_vec, _reward_component(rewards, p))
            end

        elseif nk == Kernel.CHANCE
            outcomes = _materialize(Enumerative.chance_outcomes(game, state))
            a0 = length(builder.child) + 1

            @inbounds for k in eachindex(outcomes)
                event, pr = outcomes[k]
                ns, _, _ = Kernel.step(game, state, Kernel.ChanceOutcome(event))
                nsid = _add_tree_state!(builder, ns, max_nodes)
                push!(builder.child, nsid)
                push!(builder.slot_label, convert(LT, event))
                push!(builder.action_id_within_infoset, 0)
                push!(builder.chance_prob, Float64(pr))
            end

            push!(builder.node_kind, TabularExtensiveTrees.NODE_CHANCE)
            push!(builder.node_player, 0)
            push!(builder.node_infoset, 0)
            push!(builder.node_first, a0)
            push!(builder.node_len, length(outcomes))
            push!(builder.node_active_first, 1)
            push!(builder.node_active_len, 0)
            push!(builder.reward_first, 0)

        elseif nk == Kernel.DECISION
            p = Kernel.current_player(game, state)
            info = convert(IT, ExtensiveFormInfosets.infoset(game, state, p))

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

            _ensure_infoset_action_labels!(builder, infoset_id, acts, LT)

            @inbounds for local_aid in 1:nA
                a = acts[local_aid]
                ns, _, _ = Kernel.step(game, state, a)
                nsid = _add_tree_state!(builder, ns, max_nodes)
                push!(builder.child, nsid)
                push!(builder.slot_label, convert(LT, a))
                push!(builder.action_id_within_infoset, local_aid)
                push!(builder.chance_prob, 0.0)
            end

            push!(builder.node_kind, TabularExtensiveTrees.NODE_DECISION)
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
                throw(ArgumentError("compile_extensive_tree_typed encountered a simultaneous node. Pass expand_simultaneous=true for analysis/diagnostic use."))

            aps = _materialize(Kernel.active_players(game, state))
            af0 = length(builder.active_player_ids) + 1
            append!(builder.active_player_ids, aps)

            action_lists = map(p -> _materialize(Kernel.legal_actions(game, state, p)), aps)
            a0 = length(builder.child) + 1

            for joint in Iterators.product(Tuple(action_lists)...)
                jt = Tuple(joint)
                ja = Kernel.JointAction(jt)
                ns, _, _ = Kernel.step(game, state, ja)
                nsid = _add_tree_state!(builder, ns, max_nodes)
                push!(builder.child, nsid)
                push!(builder.slot_label, convert(LT, jt))
                push!(builder.action_id_within_infoset, 0)
                push!(builder.chance_prob, 0.0)
            end

            push!(builder.node_kind, TabularExtensiveTrees.NODE_SIMULTANEOUS)
            push!(builder.node_player, 0)
            push!(builder.node_infoset, 0)
            push!(builder.node_first, a0)
            push!(builder.node_len, length(builder.child) - a0 + 1)
            push!(builder.node_active_first, af0)
            push!(builder.node_active_len, length(aps))
            push!(builder.reward_first, 0)

        else
            throw(ArgumentError("Unsupported node kind in compile_extensive_tree_typed."))
        end
    end

    return _finalize_tree(builder, infoset_enc, N, saw_simultaneous, LT)
end

function compile_extensive_graph_typed(game::Kernel.AbstractGame,
                                       root_state,
                                       ::Type{IT},
                                       ::Type{LT};
                                       max_nodes::Int = 1_000_000,
                                       state_hint::Int = 4096,
                                       edge_hint::Int = 4096,
                                       expand_simultaneous::Bool = false) where {IT,LT}
    N = Kernel.num_players(game)
    S = typeof(root_state)

    builder = _CompiledExtensiveBuilder(S, LT;
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
            push!(builder.node_kind, TabularExtensiveGraphs.NODE_TERMINAL)
            push!(builder.node_player, 0)
            push!(builder.node_infoset, 0)
            push!(builder.node_first, 1)
            push!(builder.node_len, 0)
            push!(builder.node_active_first, 1)
            push!(builder.node_active_len, 0)

            push!(builder.reward_first, length(builder.terminal_payoff_vec) + 1)
            rewards = Enumerative.terminal_payoffs(game, state)
            @inbounds for p in 1:N
                push!(builder.terminal_payoff_vec, _reward_component(rewards, p))
            end

        elseif nk == Kernel.CHANCE
            outcomes = _materialize(Enumerative.chance_outcomes(game, state))
            a0 = length(builder.child) + 1

            @inbounds for k in eachindex(outcomes)
                event, pr = outcomes[k]
                ns, _, _ = Kernel.step(game, state, Kernel.ChanceOutcome(event))
                nsid = _get_or_add_graph_state!(builder, node_enc, ns, max_nodes)
                push!(builder.child, nsid)
                push!(builder.slot_label, convert(LT, event))
                push!(builder.action_id_within_infoset, 0)
                push!(builder.chance_prob, Float64(pr))
            end

            push!(builder.node_kind, TabularExtensiveGraphs.NODE_CHANCE)
            push!(builder.node_player, 0)
            push!(builder.node_infoset, 0)
            push!(builder.node_first, a0)
            push!(builder.node_len, length(outcomes))
            push!(builder.node_active_first, 1)
            push!(builder.node_active_len, 0)
            push!(builder.reward_first, 0)

        elseif nk == Kernel.DECISION
            p = Kernel.current_player(game, state)
            info = convert(IT, ExtensiveFormInfosets.infoset(game, state, p))

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

            _ensure_infoset_action_labels!(builder, infoset_id, acts, LT)

            @inbounds for local_aid in 1:nA
                a = acts[local_aid]
                ns, _, _ = Kernel.step(game, state, a)
                nsid = _get_or_add_graph_state!(builder, node_enc, ns, max_nodes)
                push!(builder.child, nsid)
                push!(builder.slot_label, convert(LT, a))
                push!(builder.action_id_within_infoset, local_aid)
                push!(builder.chance_prob, 0.0)
            end

            push!(builder.node_kind, TabularExtensiveGraphs.NODE_DECISION)
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
                throw(ArgumentError("compile_extensive_graph_typed encountered a simultaneous node. Pass expand_simultaneous=true for analysis/diagnostic use."))

            aps = _materialize(Kernel.active_players(game, state))
            af0 = length(builder.active_player_ids) + 1
            append!(builder.active_player_ids, aps)

            action_lists = map(p -> _materialize(Kernel.legal_actions(game, state, p)), aps)
            a0 = length(builder.child) + 1

            for joint in Iterators.product(Tuple(action_lists)...)
                jt = Tuple(joint)
                ja = Kernel.JointAction(jt)
                ns, _, _ = Kernel.step(game, state, ja)
                nsid = _get_or_add_graph_state!(builder, node_enc, ns, max_nodes)
                push!(builder.child, nsid)
                push!(builder.slot_label, convert(LT, jt))
                push!(builder.action_id_within_infoset, 0)
                push!(builder.chance_prob, 0.0)
            end

            push!(builder.node_kind, TabularExtensiveGraphs.NODE_SIMULTANEOUS)
            push!(builder.node_player, 0)
            push!(builder.node_infoset, 0)
            push!(builder.node_first, a0)
            push!(builder.node_len, length(builder.child) - a0 + 1)
            push!(builder.node_active_first, af0)
            push!(builder.node_active_len, length(aps))
            push!(builder.reward_first, 0)

        else
            throw(ArgumentError("Unsupported node kind in compile_extensive_graph_typed."))
        end
    end

    return _finalize_graph(builder, node_enc, infoset_enc, N, saw_simultaneous, LT)
end

function compile_extensive_tree(game::Kernel.AbstractGame,
                                root_state;
                                max_nodes::Int = 1_000_000,
                                infoset_type::Type = Any,
                                label_type::Type = Any,
                                state_hint::Int = 4096,
                                edge_hint::Int = 4096,
                                expand_simultaneous::Bool = false)
    return compile_extensive_tree_typed(game, root_state, infoset_type, label_type;
                                        max_nodes = max_nodes,
                                        state_hint = state_hint,
                                        edge_hint = edge_hint,
                                        expand_simultaneous = expand_simultaneous)
end

function compile_extensive_graph(game::Kernel.AbstractGame,
                                 root_state;
                                 max_nodes::Int = 1_000_000,
                                 infoset_type::Type = Any,
                                 label_type::Type = Any,
                                 state_hint::Int = 4096,
                                 edge_hint::Int = 4096,
                                 expand_simultaneous::Bool = false)
    return compile_extensive_graph_typed(game, root_state, infoset_type, label_type;
                                         max_nodes = max_nodes,
                                         state_hint = state_hint,
                                         edge_hint = edge_hint,
                                         expand_simultaneous = expand_simultaneous)
end

end