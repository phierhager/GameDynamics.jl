module ExtensiveFormAnalysis

using ..Analysis
using ..Contracts
using ..CompiledExtensiveModels

export best_response_value
export average_policy_value
export exploitability_two_player_zero_sum
export ExtensiveEvaluationWorkspace
export ensure_workspace!

mutable struct ExtensiveEvaluationWorkspace
    vals::Vector{Float64}
    node_stack::Vector{Int}
    expanded_stack::Vector{UInt8}
end

function ExtensiveEvaluationWorkspace(model::CompiledExtensiveModels.CompiledExtensiveGame)
    return ExtensiveEvaluationWorkspace(
        zeros(Float64, model.n_nodes),
        Int[],
        UInt8[],
    )
end

function ensure_workspace!(ws::ExtensiveEvaluationWorkspace,
                           model::CompiledExtensiveModels.CompiledExtensiveGame)
    Contracts.supports_analysis(model) ||
        throw(ArgumentError("This compiled extensive-form model does not support analysis."))
    length(ws.vals) == model.n_nodes || resize!(ws.vals, model.n_nodes)
    empty!(ws.node_stack)
    empty!(ws.expanded_stack)
    return ws
end

@inline function _terminal_utility(model::CompiledExtensiveModels.CompiledExtensiveGame,
                                   node::Int,
                                   player::Int)
    off = model.reward_first[node]
    return model.terminal_payoffs[off + player - 1]
end

@inline function _push!(ws::ExtensiveEvaluationWorkspace, node::Int, expanded::UInt8)
    push!(ws.node_stack, node)
    push!(ws.expanded_stack, expanded)
    return nothing
end

function _policy_value_stack!(ws::ExtensiveEvaluationWorkspace,
                              model::CompiledExtensiveModels.CompiledExtensiveGame,
                              policies,
                              player::Int)
    ensure_workspace!(ws, model)
    vals = ws.vals
    fill!(vals, 0.0)

    _push!(ws, model.root_node, 0x00)

    while !isempty(ws.node_stack)
        node = pop!(ws.node_stack)
        expanded = pop!(ws.expanded_stack)
        nk = model.node_kind[node]

        if expanded == 0x00
            if nk == CompiledExtensiveModels.NODE_TERMINAL
                vals[node] = _terminal_utility(model, node, player)
            else
                _push!(ws, node, 0x01)
                a0 = model.node_first[node]
                len = model.node_len[node]
                @inbounds for k in (len - 1):-1:0
                    _push!(ws, model.child[a0 + k], 0x00)
                end
            end
        else
            a0 = model.node_first[node]
            len = model.node_len[node]

            if nk == CompiledExtensiveModels.NODE_CHANCE
                acc = 0.0
                @inbounds for k in 0:(len - 1)
                    slot = a0 + k
                    acc += model.chance_prob[slot] * vals[model.child[slot]]
                end
                vals[node] = acc

            elseif nk == CompiledExtensiveModels.NODE_SIMULTANEOUS
                acc = 0.0
                invlen = 1.0 / len
                @inbounds for k in 0:(len - 1)
                    acc += invlen * vals[model.child[a0 + k]]
                end
                vals[node] = acc

            else
                infoset = model.node_infoset[node]
                σ = policies[infoset]
                acc = 0.0
                @inbounds for a in 1:len
                    acc += σ[a] * vals[model.child[a0 + a - 1]]
                end
                vals[node] = acc
            end
        end
    end

    return vals[model.root_node]
end

function _best_response_value_stack!(ws::ExtensiveEvaluationWorkspace,
                                     model::CompiledExtensiveModels.CompiledExtensiveGame,
                                     policies,
                                     br_player::Int)
    ensure_workspace!(ws, model)
    vals = ws.vals
    fill!(vals, 0.0)

    _push!(ws, model.root_node, 0x00)

    while !isempty(ws.node_stack)
        node = pop!(ws.node_stack)
        expanded = pop!(ws.expanded_stack)
        nk = model.node_kind[node]

        if expanded == 0x00
            if nk == CompiledExtensiveModels.NODE_TERMINAL
                vals[node] = _terminal_utility(model, node, br_player)
            else
                _push!(ws, node, 0x01)
                a0 = model.node_first[node]
                len = model.node_len[node]
                @inbounds for k in (len - 1):-1:0
                    _push!(ws, model.child[a0 + k], 0x00)
                end
            end
        else
            a0 = model.node_first[node]
            len = model.node_len[node]

            if nk == CompiledExtensiveModels.NODE_CHANCE
                acc = 0.0
                @inbounds for k in 0:(len - 1)
                    slot = a0 + k
                    acc += model.chance_prob[slot] * vals[model.child[slot]]
                end
                vals[node] = acc

            elseif nk == CompiledExtensiveModels.NODE_SIMULTANEOUS
                aps = CompiledExtensiveModels.active_players(model, node)
                if br_player in aps
                    best = -Inf
                    @inbounds for k in 0:(len - 1)
                        best = max(best, vals[model.child[a0 + k]])
                    end
                    vals[node] = best
                else
                    acc = 0.0
                    invlen = 1.0 / len
                    @inbounds for k in 0:(len - 1)
                        acc += invlen * vals[model.child[a0 + k]]
                    end
                    vals[node] = acc
                end

            else
                pl = model.node_player[node]
                if pl == br_player
                    best = -Inf
                    @inbounds for a in 1:len
                        best = max(best, vals[model.child[a0 + a - 1]])
                    end
                    vals[node] = best
                else
                    infoset = model.node_infoset[node]
                    σ = policies[infoset]
                    acc = 0.0
                    @inbounds for a in 1:len
                        acc += σ[a] * vals[model.child[a0 + a - 1]]
                    end
                    vals[node] = acc
                end
            end
        end
    end

    return vals[model.root_node]
end

function average_policy_value(model::CompiledExtensiveModels.CompiledExtensiveGame,
                              policies;
                              workspace::ExtensiveEvaluationWorkspace = ExtensiveEvaluationWorkspace(model))
    return ntuple(p -> _policy_value_stack!(workspace, model, policies, p), model.n_players)
end

function best_response_value(model::CompiledExtensiveModels.CompiledExtensiveGame,
                             policies,
                             player::Int;
                             workspace::ExtensiveEvaluationWorkspace = ExtensiveEvaluationWorkspace(model))
    return _best_response_value_stack!(workspace, model, policies, player)
end

function exploitability_two_player_zero_sum(model::CompiledExtensiveModels.CompiledExtensiveGame,
                                            policies;
                                            workspace::ExtensiveEvaluationWorkspace = ExtensiveEvaluationWorkspace(model))
    model.n_players == 2 || throw(ArgumentError("exploitability_two_player_zero_sum requires a 2-player game."))
    v = average_policy_value(model, policies; workspace = workspace)
    br1 = best_response_value(model, policies, 1; workspace = workspace)
    br2 = best_response_value(model, policies, 2; workspace = workspace)
    return (br1 - v[1]) + (br2 - v[2])
end

Analysis.analysis_family(::Module) = :extensive_form

end