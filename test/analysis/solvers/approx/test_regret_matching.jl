using Test

const ApproxSolverCommon = TestHarness.ApproxSolverCommon
const RegretMatchingSolvers = TestHarness.RegretMatchingSolvers
const Fixtures = TestFixtures

@testset "solvers/approx/regret_matching.jl" begin
    @testset "workspace initialization is uniform and zeroed" begin
        # Arrange
        game = Fixtures.matching_pennies_game()

        # Act
        ws = RegretMatchingSolvers.RegretMatchingWorkspace(game)

        # Assert
        @test ws.strategy1 == [0.5, 0.5]
        @test ws.strategy2 == [0.5, 0.5]
        @test ws.regret1 == [0.0, 0.0]
        @test ws.regret2 == [0.0, 0.0]
        @test ws.strat_sum1 == [0.0, 0.0]
        @test ws.strat_sum2 == [0.0, 0.0]
    end

    @testset "reset! restores the initial workspace state" begin
        # Arrange
        game = Fixtures.matching_pennies_game()
        ws = RegretMatchingSolvers.RegretMatchingWorkspace(game)
        ws.regret1 .= [1.0, -3.0]
        ws.regret2 .= [2.0, 5.0]
        ws.strat_sum1 .= [3.0, 4.0]
        ws.strat_sum2 .= [8.0, 1.0]
        ws.strategy1 .= [1.0, 0.0]
        ws.strategy2 .= [0.0, 1.0]

        # Act
        out = RegretMatchingSolvers.reset!(ws)

        # Assert
        @test out === ws
        @test ws.strategy1 == [0.5, 0.5]
        @test ws.strategy2 == [0.5, 0.5]
        @test ws.regret1 == [0.0, 0.0]
        @test ws.regret2 == [0.0, 0.0]
        @test ws.strat_sum1 == [0.0, 0.0]
        @test ws.strat_sum2 == [0.0, 0.0]
    end

    @testset "current_strategy! normalizes positive regrets and falls back to uniform" begin
        # Arrange
        dest = zeros(3)

        # Act
        normalized = RegretMatchingSolvers.current_strategy!(dest, [2.0, -1.0, 1.0])
        uniform = RegretMatchingSolvers.current_strategy([-1.0, 0.0, -4.0])

        # Assert
        @test normalized === dest
        @test dest == [2/3, 0.0, 1/3]
        @test uniform == fill(1/3, 3)
        @test_throws ArgumentError RegretMatchingSolvers.current_strategy!(zeros(2), [1.0, 2.0, 3.0])
    end

    @testset "average_strategy! normalizes sums and falls back to uniform" begin
        # Arrange
        dest = zeros(2)

        # Act
        avg = RegretMatchingSolvers.average_strategy!(dest, [3.0, 1.0])
        uniform = RegretMatchingSolvers.average_strategy([0.0, 0.0])

        # Assert
        @test avg === dest
        @test dest == [0.75, 0.25]
        @test uniform == [0.5, 0.5]
        @test_throws ArgumentError RegretMatchingSolvers.average_strategy!(zeros(1), [1.0, 2.0])
    end

    @testset "regret_matching! accumulates valid average strategies" begin
        # Arrange
        game = Fixtures.matching_pennies_game()
        ws = RegretMatchingSolvers.RegretMatchingWorkspace(game)

        # Act
        RegretMatchingSolvers.regret_matching!(game, ws; n_iter = 500)
        avg1 = RegretMatchingSolvers.average_strategy(ws.strat_sum1)
        avg2 = RegretMatchingSolvers.average_strategy(ws.strat_sum2)
        cur1 = RegretMatchingSolvers.current_strategy(ws.regret1)
        cur2 = RegretMatchingSolvers.current_strategy(ws.regret2)

        # Assert
        @test all(x -> x >= 0.0, avg1)
        @test all(x -> x >= 0.0, avg2)
        @test isapprox(sum(avg1), 1.0; atol = 1e-12)
        @test isapprox(sum(avg2), 1.0; atol = 1e-12)
        @test isapprox(sum(cur1), 1.0; atol = 1e-12)
        @test isapprox(sum(cur2), 1.0; atol = 1e-12)
    end

    @testset "regret_matching constructor helper resets before running" begin
        # Arrange
        game = Fixtures.matching_pennies_game()
        ws = RegretMatchingSolvers.RegretMatchingWorkspace(game)
        ws.regret1 .= [100.0, -50.0]
        ws.strategy1 .= [1.0, 0.0]

        # Act
        out = RegretMatchingSolvers.regret_matching(game; n_iter = 0, workspace = ws)

        # Assert
        @test out === ws
        @test ws.strategy1 == [0.5, 0.5]
        @test ws.regret1 == [0.0, 0.0]
    end

    @testset "ApproxSolverCommon policy adapters dispatch correctly" begin
        # Arrange
        game = Fixtures.matching_pennies_game()
        ws = RegretMatchingSolvers.regret_matching(game; n_iter = 10)
        avg_dest = zeros(2)
        cur_dest = zeros(2)

        # Act
        avg1 = ApproxSolverCommon.average_policy!(avg_dest, ws, 1)
        cur2 = ApproxSolverCommon.current_policy!(cur_dest, ws, 2)

        # Assert
        @test avg1 === avg_dest
        @test cur2 === cur_dest
        @test isapprox(sum(avg_dest), 1.0; atol = 1e-12)
        @test isapprox(sum(cur_dest), 1.0; atol = 1e-12)
        @test_throws ArgumentError ApproxSolverCommon.average_policy!(zeros(2), ws, 3)
        @test_throws ArgumentError ApproxSolverCommon.current_policy!(zeros(2), ws, 3)
    end
end
