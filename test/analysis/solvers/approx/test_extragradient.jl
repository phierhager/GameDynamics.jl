using Test

const ApproxSolverCommon = TestHarness.ApproxSolverCommon
const ExtragradientSolvers = TestHarness.ExtragradientSolvers
const Fixtures = TestFixtures

@testset "solvers/approx/extragradient.jl" begin
    @testset "workspace initialization is uniform and zeroed" begin
        # Arrange
        game = Fixtures.matching_pennies_game()

        # Act
        ws = ExtragradientSolvers.ExtragradientWorkspace(game)

        # Assert
        @test ws.x == [0.5, 0.5]
        @test ws.y == [0.5, 0.5]
        @test ws.sumx == [0.0, 0.0]
        @test ws.sumy == [0.0, 0.0]
    end

    @testset "reset! restores the initial workspace state" begin
        # Arrange
        game = Fixtures.matching_pennies_game()
        ws = ExtragradientSolvers.ExtragradientWorkspace(game)
        ws.x .= [1.0, 0.0]
        ws.y .= [0.0, 1.0]
        ws.sumx .= [5.0, 6.0]
        ws.sumy .= [7.0, 8.0]

        # Act
        out = ExtragradientSolvers.reset!(ws)

        # Assert
        @test out === ws
        @test ws.x == [0.5, 0.5]
        @test ws.y == [0.5, 0.5]
        @test ws.sumx == [0.0, 0.0]
        @test ws.sumy == [0.0, 0.0]
    end

    @testset "projected_simplex! produces nonnegative vectors summing to one" begin
        # Arrange
        x = [0.2, -0.5, 1.7]
        ws = ExtragradientSolvers.SimplexProjectionWorkspace(3)
        already_simplex = [0.1, 0.2, 0.7]

        # Act
        projected = ExtragradientSolvers.projected_simplex!(x, ws)
        unchanged = ExtragradientSolvers.projected_simplex!(already_simplex, ws)

        # Assert
        @test projected === x
        @test all(v -> v >= 0.0, x)
        @test isapprox(sum(x), 1.0; atol = 1e-12)
        @test unchanged ≈ [0.1, 0.2, 0.7]
    end

    @testset "extragradient_zero_sum! keeps current and average policies valid" begin
        # Arrange
        game = Fixtures.matching_pennies_game()
        ws = ExtragradientSolvers.ExtragradientWorkspace(game)

        # Act
        ExtragradientSolvers.extragradient_zero_sum!(game, ws; n_iter = 250, η = 0.1)
        avg1 = ApproxSolverCommon.average_policy!(zeros(2), ws, 1)
        avg2 = ApproxSolverCommon.average_policy!(zeros(2), ws, 2)
        cur1 = ApproxSolverCommon.current_policy!(zeros(2), ws, 1)
        cur2 = ApproxSolverCommon.current_policy!(zeros(2), ws, 2)

        # Assert
        @test all(v -> v >= 0.0, avg1)
        @test all(v -> v >= 0.0, avg2)
        @test all(v -> v >= 0.0, cur1)
        @test all(v -> v >= 0.0, cur2)
        @test isapprox(sum(avg1), 1.0; atol = 1e-12)
        @test isapprox(sum(avg2), 1.0; atol = 1e-12)
        @test isapprox(sum(cur1), 1.0; atol = 1e-12)
        @test isapprox(sum(cur2), 1.0; atol = 1e-12)
    end

    @testset "extragradient_zero_sum helper resets before running" begin
        # Arrange
        game = Fixtures.matching_pennies_game()
        ws = ExtragradientSolvers.ExtragradientWorkspace(game)
        ws.x .= [1.0, 0.0]
        ws.y .= [0.0, 1.0]

        # Act
        x, y = ExtragradientSolvers.extragradient_zero_sum(game; n_iter = 0, workspace = ws)

        # Assert
        @test x == [0.5, 0.5]
        @test y == [0.5, 0.5]
    end

    @testset "average_policy! falls back to current policy when no averages were accumulated" begin
        # Arrange
        game = Fixtures.matching_pennies_game()
        ws = ExtragradientSolvers.ExtragradientWorkspace(game)
        ws.x .= [0.8, 0.2]
        ws.y .= [0.3, 0.7]

        # Act
        avg1 = ApproxSolverCommon.average_policy!(zeros(2), ws, 1)
        avg2 = ApproxSolverCommon.average_policy!(zeros(2), ws, 2)
        cur1 = ApproxSolverCommon.current_policy!(zeros(2), ws, 1)
        cur2 = ApproxSolverCommon.current_policy!(zeros(2), ws, 2)

        # Assert
        @test avg1 == [0.8, 0.2]
        @test avg2 == [0.3, 0.7]
        @test cur1 == [0.8, 0.2]
        @test cur2 == [0.3, 0.7]
        @test_throws ArgumentError ApproxSolverCommon.average_policy!(zeros(2), ws, 3)
        @test_throws ArgumentError ApproxSolverCommon.current_policy!(zeros(2), ws, 3)
    end
end
