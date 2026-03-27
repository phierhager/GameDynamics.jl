using Test

const TabularMatrixGames = TestHarness.TabularMatrixGames
const Fixtures = TestFixtures

@testset "tabular/matrix_game.jl" begin
    @testset "TabularMatrixGame stores shape and exposes accessors" begin
        # Arrange
        game = Fixtures.matching_pennies_game()

        # Act / Assert
        @test size(game.payoff_p1) == (2, 2)
        @test size(game.payoff_p2) == (2, 2)
        @test TabularMatrixGames.n_actions_p1(game) == 2
        @test TabularMatrixGames.n_actions_p2(game) == 2
    end
end
