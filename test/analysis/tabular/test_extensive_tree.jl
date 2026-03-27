using Test

const TabularExtensiveTrees = TestHarness.TabularExtensiveTrees
const TabularTraits = TestHarness.TabularTraits
const Fixtures = TestFixtures

@testset "tabular/extensive_tree.jl" begin
    @testset "decision-tree helpers expose structural slices" begin
        # Arrange
        tree = Fixtures.simple_tree()

        # Act
        arange = TabularExtensiveTrees.action_range(tree, 1)
        irange = TabularExtensiveTrees.infoset_range(tree, 1)
        labels = TabularExtensiveTrees.infoset_action_labels(tree, 1)
        active = TabularExtensiveTrees.active_players(tree, 1)

        # Assert
        @test TabularExtensiveTrees.n_nodes(tree) == 2
        @test TabularExtensiveTrees.n_infosets(tree) == 1
        @test TabularExtensiveTrees.node_action_count(tree, 1) == 1
        @test TabularExtensiveTrees.infoset_player(tree, 1) == 1
        @test TabularExtensiveTrees.node_infoset(tree, 1) == 1
        @test TabularExtensiveTrees.node_player(tree, 1) == 1
        @test arange == 1:1
        @test irange == 1:1
        @test collect(labels) == [:a]
        @test active == ()
        @test !TabularExtensiveTrees.has_simultaneous_nodes(tree)
        @test TabularTraits.is_tree_model(tree)
    end

    @testset "chance and simultaneous helpers expose probabilities and active players" begin
        # Arrange
        chance_tree = Fixtures.chance_tree()
        simultaneous_tree = Fixtures.simultaneous_tree()

        # Act
        probs = collect(TabularExtensiveTrees.chance_probabilities(chance_tree, 1))
        active = TabularExtensiveTrees.active_players(simultaneous_tree, 1)

        # Assert
        @test probs == [0.25, 0.75]
        @test active == (1, 2)
        @test TabularExtensiveTrees.has_simultaneous_nodes(simultaneous_tree)
    end
end
