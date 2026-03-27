module TabularValidation

using ..TabularTraits
using ..TabularMatrixGames
using ..TabularMDPs
using ..TabularMarkovGames
using ..TabularExtensiveTrees
using ..TabularExtensiveGraphs

export ValidationIssue
export ValidationReport
export validate_tabular_model
export validate_matrix_game
export validate_mdp
export validate_markov_game
export validate_extensive_tree
export validate_extensive_graph
export require_valid_tabular_model

struct ValidationIssue
    ok::Bool
    message::String
end

struct ValidationReport
    valid::Bool
    issues::Vector{ValidationIssue}
end

@inline _ok(msg::AbstractString) = ValidationIssue(true, String(msg))
@inline _bad(msg::AbstractString) = ValidationIssue(false, String(msg))

Base.show(io::IO, rep::ValidationReport) =
    print(io, "ValidationReport(valid=", rep.valid, ", issues=", length(rep.issues), ")")

@inline _all_ok(issues::Vector{ValidationIssue}) = all(x -> x.ok, issues)

function _require_nonnegative_probs(probs::AbstractVector{Float64}, label::AbstractString)
    issues = ValidationIssue[]
    @inbounds for i in eachindex(probs)
        probs[i] >= 0.0 || push!(issues, _bad("$label has negative probability at slot $i."))
    end
    return issues
end

function validate_matrix_game(g::TabularMatrixGames.TabularMatrixGame)
    issues = ValidationIssue[]

    size(g.payoff_p1) == size(g.payoff_p2) ||
        push!(issues, _bad("Player payoff matrices must have identical shape."))

    size(g.payoff_p1, 1) == g.n_actions_p1 ||
        push!(issues, _bad("n_actions_p1 does not match payoff_p1 row count."))

    size(g.payoff_p1, 2) == g.n_actions_p2 ||
        push!(issues, _bad("n_actions_p2 does not match payoff_p1 column count."))

    g.n_actions_p1 > 0 || push!(issues, _bad("Matrix game must have at least one action for player 1."))
    g.n_actions_p2 > 0 || push!(issues, _bad("Matrix game must have at least one action for player 2."))

    isempty(issues) && push!(issues, _ok("Matrix-game structure is valid."))
    return ValidationReport(_all_ok(issues), issues)
end

function validate_mdp(m::TabularMDPs.TabularMDP)
    issues = ValidationIssue[]

    length(m.n_actions_vec) == m.n_states ||
        push!(issues, _bad("n_actions_vec length must equal n_states."))

    length(m.action_ptr) == m.n_states + 1 ||
        push!(issues, _bad("action_ptr length must be n_states + 1."))

    total_sa = isempty(m.action_ptr) ? 0 : m.action_ptr[end] - 1
    length(m.trans_ptr) == total_sa + 1 ||
        push!(issues, _bad("trans_ptr length must be total_state_actions + 1."))

    length(m.action_label) == total_sa ||
        push!(issues, _bad("action_label length must equal total_state_actions."))

    length(m.states) == m.n_states ||
        push!(issues, _bad("states length must equal n_states."))

    length(m.next_state) == length(m.prob) == length(m.reward) ||
        push!(issues, _bad("next_state, prob, and reward must have equal length."))

    append!(issues, _require_nonnegative_probs(m.prob, "MDP transition array"))

    @inbounds for s in 1:m.n_states
        0 <= m.n_actions_vec[s] || push!(issues, _bad("State $s has negative action count."))

        a0 = m.action_ptr[s]
        a1 = m.action_ptr[s + 1] - 1
        expected = m.n_actions_vec[s]
        actual = max(a1 - a0 + 1, 0)

        expected == actual || push!(issues, _bad("State $s action count mismatch."))
    end

    @inbounds for sa in 1:total_sa
        t0 = m.trans_ptr[sa]
        t1 = m.trans_ptr[sa + 1] - 1
        if t1 >= t0
            sprob = 0.0
            for t in t0:t1
                ns = m.next_state[t]
                1 <= ns <= m.n_states || push!(issues, _bad("Transition at slot $t points to invalid next state $ns."))
                sprob += m.prob[t]
            end
            isapprox(sprob, 1.0; atol=1e-8) ||
                push!(issues, _bad("State-action slot $sa transition probabilities sum to $sprob, not 1."))
        end
    end

    all(iss -> iss.ok, issues) && push!(issues, _ok("Tabular MDP structure is valid."))
    return ValidationReport(_all_ok(issues), issues)
end

function validate_markov_game(m::TabularMarkovGames.TabularZeroSumMarkovGame)
    issues = ValidationIssue[]

    length(m.n_actions_p1) == m.n_states ||
        push!(issues, _bad("n_actions_p1 length must equal n_states."))

    length(m.n_actions_p2) == m.n_states ||
        push!(issues, _bad("n_actions_p2 length must equal n_states."))

    length(m.pair_ptr) == m.n_states + 1 ||
        push!(issues, _bad("pair_ptr length must be n_states + 1."))

    total_pairs = isempty(m.pair_ptr) ? 0 : m.pair_ptr[end] - 1
    length(m.trans_ptr) == total_pairs + 1 ||
        push!(issues, _bad("trans_ptr length must be total_joint_actions + 1."))

    length(m.states) == m.n_states ||
        push!(issues, _bad("states length must equal n_states."))

    length(m.legal_actions_p1) == m.n_states ||
        push!(issues, _bad("legal_actions_p1 length must equal n_states."))

    length(m.legal_actions_p2) == m.n_states ||
        push!(issues, _bad("legal_actions_p2 length must equal n_states."))

    length(m.next_state) == length(m.prob) == length(m.reward) ||
        push!(issues, _bad("next_state, prob, and reward must have equal length."))

    append!(issues, _require_nonnegative_probs(m.prob, "Markov-game transition array"))

    @inbounds for s in 1:m.n_states
        length(m.legal_actions_p1[s]) == m.n_actions_p1[s] ||
            push!(issues, _bad("State $s player-1 legal-action length mismatch."))

        length(m.legal_actions_p2[s]) == m.n_actions_p2[s] ||
            push!(issues, _bad("State $s player-2 legal-action length mismatch."))

        p0 = m.pair_ptr[s]
        p1 = m.pair_ptr[s + 1] - 1
        actual = max(p1 - p0 + 1, 0)
        expected = m.n_actions_p1[s] * m.n_actions_p2[s]

        expected == actual || push!(issues, _bad("State $s joint-action count mismatch."))
    end

    @inbounds for pair in 1:total_pairs
        t0 = m.trans_ptr[pair]
        t1 = m.trans_ptr[pair + 1] - 1
        if t1 >= t0
            sprob = 0.0
            for t in t0:t1
                ns = m.next_state[t]
                1 <= ns <= m.n_states || push!(issues, _bad("Transition at slot $t points to invalid next state $ns."))
                sprob += m.prob[t]
            end
            isapprox(sprob, 1.0; atol=1e-8) ||
                push!(issues, _bad("Joint-action slot $pair transition probabilities sum to $sprob, not 1."))
        end
    end

    all(iss -> iss.ok, issues) && push!(issues, _ok("Tabular Markov-game structure is valid."))
    return ValidationReport(_all_ok(issues), issues)
end

function _validate_extensive_common(g)
    issues = ValidationIssue[]

    length(g.node_kind) == g.n_nodes || push!(issues, _bad("node_kind length must equal n_nodes."))
    length(g.node_player) == g.n_nodes || push!(issues, _bad("node_player length must equal n_nodes."))
    length(g.node_infoset) == g.n_nodes || push!(issues, _bad("node_infoset length must equal n_nodes."))
    length(g.node_first) == g.n_nodes || push!(issues, _bad("node_first length must equal n_nodes."))
    length(g.node_len) == g.n_nodes || push!(issues, _bad("node_len length must equal n_nodes."))
    length(g.reward_first) == g.n_nodes || push!(issues, _bad("reward_first length must equal n_nodes."))

    length(g.infoset_player) == g.n_infosets || push!(issues, _bad("infoset_player length must equal n_infosets."))
    length(g.infoset_num_actions) == g.n_infosets || push!(issues, _bad("infoset_num_actions length must equal n_infosets."))
    length(g.infoset_offset) == g.n_infosets + 1 || push!(issues, _bad("infoset_offset length must be n_infosets + 1."))

    total_infoset_actions = isempty(g.infoset_offset) ? 0 : g.infoset_offset[end] - 1
    length(g.infoset_action_label) == total_infoset_actions ||
        push!(issues, _bad("infoset_action_label length mismatch."))

    length(g.node_active_first) == g.n_nodes || push!(issues, _bad("node_active_first length must equal n_nodes."))
    length(g.node_active_len) == g.n_nodes || push!(issues, _bad("node_active_len length must equal n_nodes."))

    @inbounds for node in 1:g.n_nodes
        nk = g.node_kind[node]
        a0 = g.node_first[node]
        len = g.node_len[node]
        len >= 0 || push!(issues, _bad("Node $node has negative node_len."))

        if len > 0
            a1 = a0 + len - 1
            1 <= a0 <= length(g.child) || push!(issues, _bad("Node $node has invalid node_first."))
            a1 <= length(g.child) || push!(issues, _bad("Node $node child range exceeds child array length."))
        end

        if nk == TabularExtensiveTrees.NODE_TERMINAL || nk == TabularExtensiveGraphs.NODE_TERMINAL
            r0 = g.reward_first[node]
            1 <= r0 <= length(g.terminal_payoffs) || push!(issues, _bad("Terminal node $node has invalid reward_first."))
            r1 = r0 + g.n_players - 1
            r1 <= length(g.terminal_payoffs) || push!(issues, _bad("Terminal node $node reward slice exceeds terminal_payoffs."))
        end

        if nk == TabularExtensiveTrees.NODE_DECISION || nk == TabularExtensiveGraphs.NODE_DECISION
            info = g.node_infoset[node]
            1 <= info <= g.n_infosets || push!(issues, _bad("Decision node $node has invalid infoset id $info."))
            p = g.node_player[node]
            1 <= p <= g.n_players || push!(issues, _bad("Decision node $node has invalid acting player $p."))
        end
    end

    @inbounds for i in eachindex(g.child)
        1 <= g.child[i] <= g.n_nodes || push!(issues, _bad("Child slot $i points to invalid node $(g.child[i])."))
    end

    append!(issues, _require_nonnegative_probs(g.chance_prob, "Extensive-form chance array"))

    return issues
end

function validate_extensive_tree(g::TabularExtensiveTrees.TabularExtensiveTree)
    issues = _validate_extensive_common(g)
    all(iss -> iss.ok, issues) && push!(issues, _ok("Tabular extensive tree structure is valid."))
    return ValidationReport(_all_ok(issues), issues)
end

function validate_extensive_graph(g::TabularExtensiveGraphs.TabularExtensiveGraph)
    issues = _validate_extensive_common(g)
    all(iss -> iss.ok, issues) && push!(issues, _ok("Tabular extensive graph structure is valid."))
    return ValidationReport(_all_ok(issues), issues)
end

validate_tabular_model(g::TabularMatrixGames.TabularMatrixGame) = validate_matrix_game(g)
validate_tabular_model(g::TabularMDPs.TabularMDP) = validate_mdp(g)
validate_tabular_model(g::TabularMarkovGames.TabularZeroSumMarkovGame) = validate_markov_game(g)
validate_tabular_model(g::TabularExtensiveTrees.TabularExtensiveTree) = validate_extensive_tree(g)
validate_tabular_model(g::TabularExtensiveGraphs.TabularExtensiveGraph) = validate_extensive_graph(g)

function require_valid_tabular_model(model)
    rep = validate_tabular_model(model)
    rep.valid && return model
    msgs = join((iss.message for iss in rep.issues if !iss.ok), "\n")
    throw(ArgumentError("Invalid tabular model.\n" * msgs))
end

end