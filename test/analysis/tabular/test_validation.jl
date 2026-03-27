using Test

const Encodings = TestHarness.Encodings
const TabularValidation = TestHarness.TabularValidation
const TabularMatrixGames = TestHarness.TabularMatrixGames
const Fixtures = TestFixtures

@testset "tabular/validation.jl" begin
    @testset "matrix-game validation accepts well-formed games" begin
        # Arrange
        game = Fixtures.matching_pennies_game()

        # Act
        rep = TabularValidation.validate_matrix_game(game)

        # Assert
        @test rep.valid
        @test any(iss -> iss.ok, rep.issues)
        @test TabularValidation.require_valid_tabular_model(game) === game
    end

    @testset "matrix-game validation reports shape and action-count mismatches" begin
        # Arrange
        bad = TabularMatrixGames.TabularMatrixGame(ones(2, 2), ones(2, 3), 0, 5)

        # Act
        rep = TabularValidation.validate_matrix_game(bad)
        messages = [iss.message for iss in rep.issues if !iss.ok]

        # Assert
        @test !rep.valid
        @test any(occursin("identical shape", msg) for msg in messages)
        @test any(occursin("at least one action for player 1", msg) for msg in messages)
        @test any(occursin("column count", msg) for msg in messages)
        @test_throws ArgumentError TabularValidation.require_valid_tabular_model(bad)
    end

    @testset "MDP validation checks transition structure and probabilities" begin
        # Arrange
        good = Fixtures.simple_mdp()
        bad = Fixtures.simple_mdp()
        bad = typeof(good)(
            good.n_states,
            good.n_actions_vec,
            good.action_ptr,
            good.trans_ptr,
            [99, 2],
            [-0.5, 0.9],
            good.reward,
            good.action_label,
            good.state_encoder,
            good.states,
        )

        # Act
        rep_good = TabularValidation.validate_mdp(good)
        rep_bad = TabularValidation.validate_mdp(bad)
        bad_messages = [iss.message for iss in rep_bad.issues if !iss.ok]

        # Assert
        @test rep_good.valid
        @test !rep_bad.valid
        @test any(occursin("negative probability", msg) for msg in bad_messages)
        @test any(occursin("invalid next state", msg) for msg in bad_messages)
        @test any(occursin("sum to", msg) for msg in bad_messages)
    end

    @testset "Markov-game validation checks legal actions and transition structure" begin
        # Arrange
        good = Fixtures.simple_markov_game()
        bad = Fixtures.simple_markov_game()
        bad = typeof(good)(
            good.n_states,
            good.n_actions_p1,
            good.n_actions_p2,
            [1, 2, 3],
            good.trans_ptr,
            [2, 9],
            [1.0, -1.0],
            good.reward,
            good.state_encoder,
            good.states,
            [[11, 22], Int[]],
            [[7], Int[]],
        )

        # Act
        rep_good = TabularValidation.validate_markov_game(good)
        rep_bad = TabularValidation.validate_markov_game(bad)
        bad_messages = [iss.message for iss in rep_bad.issues if !iss.ok]

        # Assert
        @test rep_good.valid
        @test !rep_bad.valid
        @test any(occursin("joint-action count mismatch", msg) for msg in bad_messages)
        @test any(occursin("negative probability", msg) for msg in bad_messages)
        @test any(occursin("invalid next state", msg) for msg in bad_messages)
    end

    @testset "extensive-tree validation catches bad child pointers and terminal reward spans" begin
        # Arrange
        good = Fixtures.simple_tree()
        bad = Fixtures.simple_tree()
        bad = typeof(good)(
            good.n_players,
            good.has_simultaneous,
            good.n_nodes,
            good.node_kind,
            good.node_player,
            good.node_infoset,
            good.node_first,
            good.node_len,
            [3],
            good.slot_label,
            good.action_id_within_infoset,
            [-0.1],
            good.n_infosets,
            good.infoset_player,
            good.infoset_num_actions,
            good.infoset_offset,
            good.infoset_action_label,
            good.node_active_first,
            good.node_active_len,
            good.active_player_ids,
            good.n_terminals,
            [0, 2],
            [1.0],
            good.infoset_encoder,
            good.root_node,
        )

        # Act
        rep_good = TabularValidation.validate_extensive_tree(good)
        rep_bad = TabularValidation.validate_extensive_tree(bad)
        bad_messages = [iss.message for iss in rep_bad.issues if !iss.ok]

        # Assert
        @test rep_good.valid
        @test !rep_bad.valid
        @test any(occursin("invalid node", msg) for msg in bad_messages)
        @test any(occursin("negative probability", msg) for msg in bad_messages)
        @test any(occursin("reward slice exceeds", msg) for msg in bad_messages)
    end

    @testset "extensive-graph validation catches bad chance and child structure" begin
        # Arrange
        good = Fixtures.simple_graph()
        bad = Fixtures.simple_graph()
        bad = typeof(good)(
            good.n_players,
            good.has_simultaneous,
            good.n_nodes,
            good.node_kind,
            good.node_player,
            good.node_infoset,
            good.node_first,
            good.node_len,
            [99],
            good.slot_label,
            good.action_id_within_infoset,
            [-0.2],
            good.n_infosets,
            good.infoset_player,
            good.infoset_num_actions,
            good.infoset_offset,
            good.infoset_action_label,
            good.node_active_first,
            good.node_active_len,
            good.active_player_ids,
            good.n_terminals,
            good.reward_first,
            good.terminal_payoffs,
            good.node_encoder,
            good.infoset_encoder,
            good.root_node,
        )

        # Act
        rep_good = TabularValidation.validate_extensive_graph(good)
        rep_bad = TabularValidation.validate_extensive_graph(bad)
        bad_messages = [iss.message for iss in rep_bad.issues if !iss.ok]

        # Assert
        @test rep_good.valid
        @test !rep_bad.valid
        @test any(occursin("invalid node", msg) for msg in bad_messages)
        @test any(occursin("negative probability", msg) for msg in bad_messages)
    end

    @testset "validate_tabular_model dispatches by model type" begin
        # Arrange
        matrix_game = Fixtures.matching_pennies_game()
        mdp = Fixtures.simple_mdp()
        markov_game = Fixtures.simple_markov_game()
        tree = Fixtures.simple_tree()
        graph = Fixtures.simple_graph()

        # Act / Assert
        @test TabularValidation.validate_tabular_model(matrix_game).valid
        @test TabularValidation.validate_tabular_model(mdp).valid
        @test TabularValidation.validate_tabular_model(markov_game).valid
        @test TabularValidation.validate_tabular_model(tree).valid
        @test TabularValidation.validate_tabular_model(graph).valid
    end
end
