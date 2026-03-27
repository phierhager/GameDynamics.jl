module CFRSolvers

using ..CompiledExtensiveModels
using ..ApproxSolverCommon

export CFRWorkspace
export CFRPlusWorkspace
export regret_matching_policy!
export copy_average_policy!
export cfr_iteration!
export cfrplus_iteration!
export run_cfr!
export run_cfrplus!
export extract_average_policy
export extract_current_policy

mutable struct CFRWorkspace
    regrets::Vector{Float64}
    strategy_sum::Vector{Float64}
    current_strategy::Vector{Float64}

    node_value::Vector{Float64}

    post_node_stack::Vector{Int}
    post_expanded_stack::Vector{UInt8}

    fwd_node_stack::Vector{Int}
    fwd_p1_stack::Vector{Float64}
    fwd_p2_stack::Vector{Float64}
    fwd_pc_stack::Vector{Float64}
end

mutable struct CFRPlusWorkspace
    base::CFRWorkspace
    iteration::Int
    averaging_delay::Int
end

function CFRWorkspace(model::CompiledExtensiveModels.CompiledExtensiveGame)
    total_actions = model.infoset_offset[end] - 1
    return CFRWorkspace(
        zeros(total_actions),
        zeros(total_actions),
        zeros(total_actions),

        zeros(model.n_nodes),

        Int[],
        UInt8[],

        Int[],
        Float64[],
        Float64[],
        Float64[],
    )
end

CFRPlusWorkspace(model::CompiledExtensiveModels.CompiledExtensiveGame; averaging_delay::Int = 0) =
    CFRPlusWorkspace(CFRWorkspace(model), 0, averaging_delay)

@inline _i0(model, infoset::Int) = model.infoset_offset[infoset]
@inline _i1(model, infoset::Int) = model.infoset_offset[infoset + 1] - 1
@inline _nA(model, infoset::Int) = model.infoset_num_actions[infoset]

@inline function _terminal_utility(model::CompiledExtensiveModels.CompiledExtensiveGame,
                                   node::Int,
                                   player::Int)
    off = model.reward_first[node]
    return model.terminal_payoffs[off + player - 1]
end

function regret_matching_policy!(ws::CFRWorkspace,
                                 model::CompiledExtensiveModels.CompiledExtensiveGame,
                                 infoset::Int)
    i0 = _i0(model, infoset)
    i1 = _i1(model, infoset)

    z = 0.0
    @inbounds for i in i0:i1
        v = max(ws.regrets[i], 0.0)
        ws.current_strategy[i] = v
        z += v
    end

    if z > 0
        invz = 1 / z
        @inbounds for i in i0:i1
            ws.current_strategy[i] *= invz
        end
    else
        v = 1 / (i1 - i0 + 1)
        @inbounds for i in i0:i1
            ws.current_strategy[i] = v
        end
    end

    return nothing
end

function _write_current_policy!(dest::AbstractVector{Float64},
                                ws::CFRWorkspace,
                                model::CompiledExtensiveModels.CompiledExtensiveGame,
                                infoset::Int)
    i0 = _i0(model, infoset)
    i1 = _i1(model, infoset)
    length(dest) == (i1 - i0 + 1) || throw(ArgumentError("Destination length mismatch."))

    z = 0.0
    @inbounds for i in i0:i1
        z += max(ws.regrets[i], 0.0)
    end

    if z > 0
        invz = 1 / z
        @inbounds for (k, i) in enumerate(i0:i1)
            dest[k] = max(ws.regrets[i], 0.0) * invz
        end
    else
        v = 1 / length(dest)
        @inbounds for k in eachindex(dest)
            dest[k] = v
        end
    end

    return dest
end

function copy_average_policy!(dest::AbstractVector{Float64},
                              ws::CFRWorkspace,
                              model::CompiledExtensiveModels.CompiledExtensiveGame,
                              infoset::Int)
    i0 = _i0(model, infoset)
    i1 = _i1(model, infoset)
    length(dest) == (i1 - i0 + 1) || throw(ArgumentError("Destination length mismatch."))

    z = 0.0
    @inbounds for i in i0:i1
        z += ws.strategy_sum[i]
    end

    if z > 0
        invz = 1 / z
        @inbounds for (k, i) in enumerate(i0:i1)
            dest[k] = ws.strategy_sum[i] * invz
        end
    else
        v = 1 / length(dest)
        @inbounds for k in eachindex(dest)
            dest[k] = v
        end
    end

    return dest
end

@inline function _prepare_policies!(ws::CFRWorkspace,
                                    model::CompiledExtensiveModels.CompiledExtensiveGame)
    @inbounds for infoset in 1:model.n_infosets
        regret_matching_policy!(ws, model, infoset)
    end
    return ws
end

@inline function _push_post!(ws::CFRWorkspace, node::Int, expanded::UInt8)
    push!(ws.post_node_stack, node)
    push!(ws.post_expanded_stack, expanded)
    return nothing
end

@inline function _push_fwd!(ws::CFRWorkspace,
                            node::Int,
                            p1::Float64,
                            p2::Float64,
                            pc::Float64)
    push!(ws.fwd_node_stack, node)
    push!(ws.fwd_p1_stack, p1)
    push!(ws.fwd_p2_stack, p2)
    push!(ws.fwd_pc_stack, pc)
    return nothing
end

function _compute_node_values_stack!(model::CompiledExtensiveModels.CompiledExtensiveGame,
                                     ws::CFRWorkspace,
                                     updating_player::Int)
    empty!(ws.post_node_stack)
    empty!(ws.post_expanded_stack)

    _push_post!(ws, model.root_node, 0x00)

    while !isempty(ws.post_node_stack)
        node = pop!(ws.post_node_stack)
        expanded = pop!(ws.post_expanded_stack)
        nk = model.node_kind[node]

        if expanded == 0x00
            if nk == CompiledExtensiveModels.NODE_TERMINAL
                ws.node_value[node] = _terminal_utility(model, node, updating_player)
            else
                _push_post!(ws, node, 0x01)
                a0 = model.node_first[node]
                len = model.node_len[node]
                @inbounds for k in (len - 1):-1:0
                    _push_post!(ws, model.child[a0 + k], 0x00)
                end
            end
        else
            a0 = model.node_first[node]
            len = model.node_len[node]

            if nk == CompiledExtensiveModels.NODE_CHANCE
                acc = 0.0
                @inbounds for k in 0:(len - 1)
                    slot = a0 + k
                    acc += model.chance_prob[slot] * ws.node_value[model.child[slot]]
                end
                ws.node_value[node] = acc

            elseif nk == CompiledExtensiveModels.NODE_SIMULTANEOUS
                throw(ArgumentError("Simultaneous nodes are not supported in CFR value computation."))

            else
                infoset = model.node_infoset[node]
                s0 = _i0(model, infoset)
                acc = 0.0
                @inbounds for a in 1:len
                    acc += ws.current_strategy[s0 + a - 1] * ws.node_value[model.child[a0 + a - 1]]
                end
                ws.node_value[node] = acc
            end
        end
    end

    return ws
end

function _update_from_values_stack!(model::CompiledExtensiveModels.CompiledExtensiveGame,
                                    ws::CFRWorkspace,
                                    updating_player::Int,
                                    avg_coeff::Float64;
                                    plus::Bool = false)
    empty!(ws.fwd_node_stack)
    empty!(ws.fwd_p1_stack)
    empty!(ws.fwd_p2_stack)
    empty!(ws.fwd_pc_stack)

    _push_fwd!(ws, model.root_node, 1.0, 1.0, 1.0)

    while !isempty(ws.fwd_node_stack)
        node = pop!(ws.fwd_node_stack)
        p1 = pop!(ws.fwd_p1_stack)
        p2 = pop!(ws.fwd_p2_stack)
        pc = pop!(ws.fwd_pc_stack)

        nk = model.node_kind[node]

        if nk == CompiledExtensiveModels.NODE_TERMINAL
            continue

        elseif nk == CompiledExtensiveModels.NODE_CHANCE
            a0 = model.node_first[node]
            len = model.node_len[node]
            @inbounds for k in (len - 1):-1:0
                slot = a0 + k
                _push_fwd!(ws, model.child[slot], p1, p2, pc * model.chance_prob[slot])
            end

        elseif nk == CompiledExtensiveModels.NODE_SIMULTANEOUS
            throw(ArgumentError("Simultaneous nodes are not supported in CFR update traversal."))

        else
            infoset = model.node_infoset[node]
            pl = model.node_player[node]
            a0 = model.node_first[node]
            len = model.node_len[node]
            s0 = _i0(model, infoset)

            node_util = ws.node_value[node]

            if pl == updating_player
                if pl == 1
                    cf_reach = p2 * pc
                    avg_reach = p1 * pc
                    @inbounds for a in 1:len
                        i = s0 + a - 1
                        child = model.child[a0 + a - 1]
                        if plus
                            ws.regrets[i] = max(ws.regrets[i] + cf_reach * (ws.node_value[child] - node_util), 0.0)
                        else
                            ws.regrets[i] += cf_reach * (ws.node_value[child] - node_util)
                        end
                        ws.strategy_sum[i] += avg_coeff * avg_reach * ws.current_strategy[i]
                    end
                    @inbounds for a in len:-1:1
                        i = s0 + a - 1
                        _push_fwd!(ws, model.child[a0 + a - 1], p1 * ws.current_strategy[i], p2, pc)
                    end
                else
                    cf_reach = p1 * pc
                    avg_reach = p2 * pc
                    @inbounds for a in 1:len
                        i = s0 + a - 1
                        child = model.child[a0 + a - 1]
                        if plus
                            ws.regrets[i] = max(ws.regrets[i] + cf_reach * (ws.node_value[child] - node_util), 0.0)
                        else
                            ws.regrets[i] += cf_reach * (ws.node_value[child] - node_util)
                        end
                        ws.strategy_sum[i] += avg_coeff * avg_reach * ws.current_strategy[i]
                    end
                    @inbounds for a in len:-1:1
                        i = s0 + a - 1
                        _push_fwd!(ws, model.child[a0 + a - 1], p1, p2 * ws.current_strategy[i], pc)
                    end
                end
            else
                if pl == 1
                    @inbounds for a in len:-1:1
                        i = s0 + a - 1
                        _push_fwd!(ws, model.child[a0 + a - 1], p1 * ws.current_strategy[i], p2, pc)
                    end
                else
                    @inbounds for a in len:-1:1
                        i = s0 + a - 1
                        _push_fwd!(ws, model.child[a0 + a - 1], p1, p2 * ws.current_strategy[i], pc)
                    end
                end
            end
        end
    end

    return ws
end

function cfr_iteration!(model::CompiledExtensiveModels.CompiledExtensiveGame,
                        ws::CFRWorkspace)
    ApproxSolverCommon.require_supported_2p_tree_model(model)
    _prepare_policies!(ws, model)

    @inbounds for p in 1:2
        _compute_node_values_stack!(model, ws, p)
        _update_from_values_stack!(model, ws, p, 1.0; plus = false)
    end

    return ws
end

function cfrplus_iteration!(model::CompiledExtensiveModels.CompiledExtensiveGame,
                            ws::CFRPlusWorkspace)
    ApproxSolverCommon.require_supported_2p_tree_model(model)

    ws.iteration += 1
    avg_coeff = max(ws.iteration - ws.averaging_delay, 0)

    _prepare_policies!(ws.base, model)

    @inbounds for p in 1:2
        _compute_node_values_stack!(model, ws.base, p)
        _update_from_values_stack!(model, ws.base, p, Float64(avg_coeff); plus = true)
    end

    return ws
end

function run_cfr!(model::CompiledExtensiveModels.CompiledExtensiveGame,
                  ws::CFRWorkspace = CFRWorkspace(model);
                  n_iter::Int = 1_000)
    for _ in 1:n_iter
        cfr_iteration!(model, ws)
    end
    return ws
end

function run_cfrplus!(model::CompiledExtensiveModels.CompiledExtensiveGame,
                      ws::CFRPlusWorkspace = CFRPlusWorkspace(model);
                      n_iter::Int = 1_000)
    for _ in 1:n_iter
        cfrplus_iteration!(model, ws)
    end
    return ws
end

function extract_average_policy(model::CompiledExtensiveModels.CompiledExtensiveGame,
                                ws::CFRWorkspace)
    out = Dict{Int,Vector{Pair{eltype(model.infoset_action_label),Float64}}}()
    for infoset in 1:model.n_infosets
        nA = _nA(model, infoset)
        probs = Vector{Float64}(undef, nA)
        copy_average_policy!(probs, ws, model, infoset)
        labs = CompiledExtensiveModels.infoset_action_labels(model, infoset)
        out[infoset] = [labs[a] => probs[a] for a in 1:nA]
    end
    return out
end

function extract_current_policy(model::CompiledExtensiveModels.CompiledExtensiveGame,
                                ws::CFRWorkspace)
    out = Dict{Int,Vector{Pair{eltype(model.infoset_action_label),Float64}}}()
    for infoset in 1:model.n_infosets
        nA = _nA(model, infoset)
        probs = Vector{Float64}(undef, nA)
        _write_current_policy!(probs, ws, model, infoset)
        labs = CompiledExtensiveModels.infoset_action_labels(model, infoset)
        out[infoset] = [labs[a] => probs[a] for a in 1:nA]
    end
    return out
end

end