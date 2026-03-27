module TabularExtensiveGraphs

using ..TabularTraits

export TabularExtensiveGraph
export NODE_TERMINAL, NODE_CHANCE, NODE_DECISION, NODE_SIMULTANEOUS
export n_nodes, n_infosets, node_action_count
export infoset_player, node_infoset, node_player, active_players
export chance_probabilities, action_range, infoset_range
export infoset_action_labels, has_simultaneous_nodes

const NODE_TERMINAL     = UInt8(0x00)
const NODE_CHANCE       = UInt8(0x01)
const NODE_DECISION     = UInt8(0x02)
const NODE_SIMULTANEOUS = UInt8(0x03)

struct TabularExtensiveGraph{NE,IE,AL} <: TabularTraits.AbstractTabularExtensiveFormModel
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

n_nodes(g::TabularExtensiveGraph) = g.n_nodes
n_infosets(g::TabularExtensiveGraph) = g.n_infosets
node_action_count(g::TabularExtensiveGraph, node::Int) = g.node_len[node]
infoset_player(g::TabularExtensiveGraph, infoset::Int) = g.infoset_player[infoset]
node_infoset(g::TabularExtensiveGraph, node::Int) = g.node_infoset[node]
node_player(g::TabularExtensiveGraph, node::Int) = g.node_player[node]
has_simultaneous_nodes(g::TabularExtensiveGraph) = g.has_simultaneous

@inline function action_range(g::TabularExtensiveGraph, node::Int)
    a0 = g.node_first[node]
    return a0:(a0 + g.node_len[node] - 1)
end

@inline function infoset_range(g::TabularExtensiveGraph, infoset::Int)
    i0 = g.infoset_offset[infoset]
    return i0:(g.infoset_offset[infoset + 1] - 1)
end

chance_probabilities(g::TabularExtensiveGraph, node::Int) =
    @view g.chance_prob[action_range(g, node)]

function active_players(g::TabularExtensiveGraph, node::Int)
    k0 = g.node_active_first[node]
    len = g.node_active_len[node]
    len == 0 && return ()
    return Tuple(g.active_player_ids[k0:(k0 + len - 1)])
end

infoset_action_labels(g::TabularExtensiveGraph, infoset::Int) =
    g.infoset_action_label[infoset_range(g, infoset)]

TabularTraits.is_graph_model(::TabularExtensiveGraph) = true
TabularTraits.supports_exact_solvers(::TabularExtensiveGraph) = true
TabularTraits.supports_approx_solvers(::TabularExtensiveGraph) = false

end