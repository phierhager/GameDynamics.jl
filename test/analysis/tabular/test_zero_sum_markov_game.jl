using Test

const TabularMarkovGames = TestHarness.TabularMarkovGames
const Fixtures = TestFixtures

@testset "tabular/zero_sum_markov_game.jl" begin
    @testset "TabularZeroSumMarkovGame exposes per-state action sizes" begin
        # Arrange
        model = Fixtures.simple_markov_game()

        # Act / Assert
        @test TabularMarkovGames.n_states(model) == 2
        @test TabularMarkovGames.n_actions(model, 1) == (2, 1)
        @test TabularMarkovGames.n_actions(model, 2) == (0, 0)
    end
end
