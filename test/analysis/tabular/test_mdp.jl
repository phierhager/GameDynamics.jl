using Test

const TabularMDPs = TestHarness.TabularMDPs
const Fixtures = TestFixtures

@testset "tabular/mdp.jl" begin
    @testset "TabularMDP accessors expose counts and labels" begin
        # Arrange
        model = Fixtures.simple_mdp()

        # Act
        labels_s1 = TabularMDPs.action_labels(model, 1)
        labels_s2 = TabularMDPs.action_labels(model, 2)

        # Assert
        @test TabularMDPs.n_states(model) == 2
        @test TabularMDPs.n_actions(model, 1) == 2
        @test TabularMDPs.n_actions(model, 2) == 0
        @test collect(labels_s1) == [10, 20]
        @test isempty(labels_s2)
    end
end
