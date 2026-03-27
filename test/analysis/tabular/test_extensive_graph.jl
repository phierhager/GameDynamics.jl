using Test

const TabularExtensiveGraphs = TestHarness.TabularExtensiveGraphs
const TabularTraits = TestHarness.TabularTraits
const Fixtures = TestFixtures

@testset "tabular/extensive_graph.jl" begin
    @testset "graph helpers expose structural slices" begin
        # Arrange
        graph = Fixtures.simple_graph()

        # Act
        arange = TabularExtensiveGraphs.action_range(graph, 1)
        irange = TabularExtensiveGraphs.infoset_range(graph, 1)
        labels = TabularExtensiveGraphs.infoset_action_labels(graph, 1)
        active = TabularExtensiveGraphs.active_players(graph, 1)

        # Assert
        @test TabularExtensiveGraphs.n_nodes(graph) == 2
        @test TabularExtensiveGraphs.n_infosets(graph) == 1
        @test TabularExtensiveGraphs.node_action_count(graph, 1) == 1
        @test TabularExtensiveGraphs.infoset_player(graph, 1) == 1
        @test TabularExtensiveGraphs.node_infoset(graph, 1) == 1
        @test TabularExtensiveGraphs.node_player(graph, 1) == 1
        @test arange == 1:1
        @test irange == 1:1
        @test collect(labels) == [:a]
        @test active == ()
        @test !TabularExtensiveGraphs.has_simultaneous_nodes(graph)
        @test TabularTraits.is_graph_model(graph)
    end

    @testset "chance graph exposes chance probabilities" begin
        # Arrange
        graph = Fixtures.chance_graph()

        # Act
        probs = collect(TabularExtensiveGraphs.chance_probabilities(graph, 1))

        # Assert
        @test probs == [0.25, 0.75]
    end
end
