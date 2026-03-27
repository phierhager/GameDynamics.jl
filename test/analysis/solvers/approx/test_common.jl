using Test

const ApproxSolverCommon = TestHarness.ApproxSolverCommon
const TabularTraits = TestHarness.TabularTraits
const Fixtures = TestFixtures

struct DummyDiagnosticsOnly <: TabularTraits.AbstractTabularModel end
TabularTraits.is_solver_grade(::DummyDiagnosticsOnly) = false
TabularTraits.model_role(::DummyDiagnosticsOnly) = :diagnostics_only

@testset "solvers/approx/common.jl" begin
    @testset "require_solver_grade_model accepts solver-grade models" begin
        # Arrange
        model = Fixtures.matching_pennies_game()

        # Act / Assert
        @test ApproxSolverCommon.require_solver_grade_model(model) === nothing
    end

    @testset "require_solver_grade_model rejects diagnostics-only models" begin
        # Arrange
        model = DummyDiagnosticsOnly()

        # Act / Assert
        @test_throws ArgumentError ApproxSolverCommon.require_solver_grade_model(model)
    end

    @testset "require_supported_2p_tree_model rejects unsupported approximate paths" begin
        # Arrange
        supported = Fixtures.simple_tree()
        unsupported = Fixtures.simultaneous_tree()

        # Act / Assert
        @test ApproxSolverCommon.require_supported_2p_tree_model(supported) === nothing
        @test_throws ArgumentError ApproxSolverCommon.require_supported_2p_tree_model(unsupported)
    end

    @testset "require_tabular_2p_matrix_game enforces positive action counts" begin
        # Arrange
        good = Fixtures.matching_pennies_game()
        bad = TestHarness.TabularMatrixGames.TabularMatrixGame(zeros(0, 1), zeros(0, 1), 0, 1)

        # Act / Assert
        @test ApproxSolverCommon.require_tabular_2p_matrix_game(good) === nothing
        @test_throws ArgumentError ApproxSolverCommon.require_tabular_2p_matrix_game(bad)
    end

    @testset "generic solver hooks throw MethodError by default" begin
        # Arrange
        dummy = IdDict()

        # Act / Assert
        @test_throws MethodError ApproxSolverCommon.reset_solver!(dummy)
        @test_throws MethodError ApproxSolverCommon.run_solver!(nothing, dummy)
        @test_throws MethodError ApproxSolverCommon.average_policy!(zeros(2), dummy, 1)
        @test_throws MethodError ApproxSolverCommon.current_policy!(zeros(2), dummy, 1)
    end
end
