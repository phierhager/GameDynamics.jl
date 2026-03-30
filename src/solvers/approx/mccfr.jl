module MCCFRSolvers

using Random
using ..TabularExtensiveTrees
using ..CFRSolvers
using ..ApproxSolverCommon

export ExternalSamplingWorkspace
export mccfr_iteration!
export mccfrplus_iteration!
export run_mccfr!
export run_mccfrplus!
export outcome_sampling_iteration!
export outcome_sampling_plus_iteration!
export run_outcome_sampling_mccfr!
export run_outcome_sampling_mccfrplus!

mutable struct ExternalSamplingWorkspace{R<:AbstractRNG}
    cfr::CFRSolvers.CFRWorkspace
    rng::R
    iteration::Int
    averaging_delay::Int
end

ExternalSamplingWorkspace(model::TabularExtensiveTrees.TabularExtensiveTree;
                          rng::R = Random.default_rng(),
                          averaging_delay::Int = 0) where {R<:AbstractRNG} =
    ExternalSamplingWorkspace{R}(CFRSolvers.CFRWorkspace(model), rng, 0, averaging_delay)

@inline _i0(model, infoset::Int) = model.infoset_offset[infoset]

@inline function _sample_index(probs::AbstractVector{<:Real}, rng::AbstractRNG)
    r = rand(rng)
    c = 0.0
    @inbounds for i in eachindex(probs)
        c += probs[i]
        if r <= c
            return i
        end
    end
    return last(eachindex(probs))
end

@inline function _terminal_utility(model::TabularExtensiveTrees.TabularExtensiveTree,
                                   node::Int,
                                   player::Int)
    off = model.reward_first[node]
    return model.terminal_payoffs[off + player - 1]
end

function _external_sampling_traverse!(model::TabularExtensiveTrees.TabularExtensiveTree,
                                      ws::ExternalSamplingWorkspace,
                                      node::Int,
                                      updating_player::Int,
                                      p1::Float64,
                                      p2::Float64,
                                      pc::Float64,
                                      avg_coeff::Float64;
                                      plus::Bool = false)
    nk = model.node_kind[node]

    if nk == TabularExtensiveTrees.NODE_TERMINAL
        return _terminal_utility(model, node, updating_player)

    elseif nk == TabularExtensiveTrees.NODE_CHANCE
        pr = TabularExtensiveTrees.chance_probabilities(model, node)
        j = _sample_index(pr, ws.rng)
        slot = model.node_first[node] + j - 1
        return _external_sampling_traverse!(model, ws, model.child[slot], updating_player, p1, p2, pc * pr[j], avg_coeff; plus = plus)

    elseif nk == TabularExtensiveTrees.NODE_SIMULTANEOUS
        throw(ArgumentError("Simultaneous nodes are not supported in MCCFR traversal."))

    else
        infoset = model.node_infoset[node]
        pl = model.node_player[node]
        CFRSolvers.regret_matching_policy!(ws.cfr, model, infoset)

        a0 = model.node_first[node]
        len = model.node_len[node]
        s0 = _i0(model, infoset)

        if pl == updating_player
            node_util = 0.0

            if pl == 1
                @inbounds for a in 1:len
                    i = s0 + a - 1
                    u = _external_sampling_traverse!(model, ws, model.child[a0 + a - 1], updating_player, p1 * ws.cfr.current_strategy[i], p2, pc, avg_coeff; plus = plus)
                    ws.cfr.node_value[a] = u
                    node_util += ws.cfr.current_strategy[i] * u
                end

                cf_reach = p2 * pc
                avg_reach = p1 * pc

                @inbounds for a in 1:len
                    i = s0 + a - 1
                    if plus
                        ws.cfr.regrets[i] = max(ws.cfr.regrets[i] + cf_reach * (ws.cfr.node_value[a] - node_util), 0.0)
                    else
                        ws.cfr.regrets[i] += cf_reach * (ws.cfr.node_value[a] - node_util)
                    end
                    ws.cfr.strategy_sum[i] += avg_coeff * avg_reach * ws.cfr.current_strategy[i]
                end
            else
                @inbounds for a in 1:len
                    i = s0 + a - 1
                    u = _external_sampling_traverse!(model, ws, model.child[a0 + a - 1], updating_player, p1, p2 * ws.cfr.current_strategy[i], pc, avg_coeff; plus = plus)
                    ws.cfr.node_value[a] = u
                    node_util += ws.cfr.current_strategy[i] * u
                end

                cf_reach = p1 * pc
                avg_reach = p2 * pc

                @inbounds for a in 1:len
                    i = s0 + a - 1
                    if plus
                        ws.cfr.regrets[i] = max(ws.cfr.regrets[i] + cf_reach * (ws.cfr.node_value[a] - node_util), 0.0)
                    else
                        ws.cfr.regrets[i] += cf_reach * (ws.cfr.node_value[a] - node_util)
                    end
                    ws.cfr.strategy_sum[i] += avg_coeff * avg_reach * ws.cfr.current_strategy[i]
                end
            end

            return node_util

        else
            probs = @view ws.cfr.current_strategy[s0:(s0 + len - 1)]
            a = _sample_index(probs, ws.rng)
            σa = probs[a]

            if pl == 1
                return _external_sampling_traverse!(model, ws, model.child[a0 + a - 1], updating_player, p1 * σa, p2, pc, avg_coeff; plus = plus)
            else
                return _external_sampling_traverse!(model, ws, model.child[a0 + a - 1], updating_player, p1, p2 * σa, pc, avg_coeff; plus = plus)
            end
        end
    end
end

function mccfr_iteration!(model::TabularExtensiveTrees.TabularExtensiveTree,
                          ws::ExternalSamplingWorkspace)
    ApproxSolverCommon.require_supported_2p_tree_model(model)

    @inbounds for p in 1:2
        _external_sampling_traverse!(model, ws, model.root_node, p, 1.0, 1.0, 1.0, 1.0; plus = false)
    end

    return ws
end

function mccfrplus_iteration!(model::TabularExtensiveTrees.TabularExtensiveTree,
                              ws::ExternalSamplingWorkspace)
    ApproxSolverCommon.require_supported_2p_tree_model(model)

    ws.iteration += 1
    avg_coeff = Float64(max(ws.iteration - ws.averaging_delay, 0))

    @inbounds for p in 1:2
        _external_sampling_traverse!(model, ws, model.root_node, p, 1.0, 1.0, 1.0, avg_coeff; plus = true)
    end

    return ws
end

function run_mccfr!(model::TabularExtensiveTrees.TabularExtensiveTree,
                    ws::ExternalSamplingWorkspace = ExternalSamplingWorkspace(model);
                    n_iter::Int = 10_000)
    ApproxSolverCommon.require_supported_2p_tree_model(model)
    for _ in 1:n_iter
        mccfr_iteration!(model, ws)
    end
    return ws
end

function run_mccfrplus!(model::TabularExtensiveTrees.TabularExtensiveTree,
                        ws::ExternalSamplingWorkspace = ExternalSamplingWorkspace(model);
                        n_iter::Int = 10_000)
    ApproxSolverCommon.require_supported_2p_tree_model(model)
    for _ in 1:n_iter
        mccfrplus_iteration!(model, ws)
    end
    return ws
end

function outcome_sampling_iteration!(model::TabularExtensiveTrees.TabularExtensiveTree,
                                     ws::ExternalSamplingWorkspace)
    ApproxSolverCommon.require_supported_2p_tree_model(model)
    throw(ArgumentError("Outcome-sampling MCCFR is temporarily disabled here. The previous implementation was not solver-grade. Re-enable only with a fully validated sampled-counterfactual update."))
end

function outcome_sampling_plus_iteration!(model::TabularExtensiveTrees.TabularExtensiveTree,
                                          ws::ExternalSamplingWorkspace)
    ApproxSolverCommon.require_supported_2p_tree_model(model)
    throw(ArgumentError("Outcome-sampling MCCFR+ is temporarily disabled here. The previous implementation was not solver-grade. Re-enable only with a fully validated sampled-counterfactual update."))
end

function run_outcome_sampling_mccfr!(model::TabularExtensiveTrees.TabularExtensiveTree,
                                     ws::ExternalSamplingWorkspace = ExternalSamplingWorkspace(model);
                                     n_iter::Int = 10_000)
    ApproxSolverCommon.require_supported_2p_tree_model(model)
    throw(ArgumentError("Outcome-sampling MCCFR is temporarily disabled here."))
end

function run_outcome_sampling_mccfrplus!(model::TabularExtensiveTrees.TabularExtensiveTree,
                                         ws::ExternalSamplingWorkspace = ExternalSamplingWorkspace(model);
                                         n_iter::Int = 10_000)
    ApproxSolverCommon.require_supported_2p_tree_model(model)
    throw(ArgumentError("Outcome-sampling MCCFR+ is temporarily disabled here."))
end

end