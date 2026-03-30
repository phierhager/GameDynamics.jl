using Test
using Random

using GameLab
using GameLab.StrategyInterface
using GameLab.LocalStrategies

@testset "StrategyInterface" begin
    @testset "trait defaults throw when not implemented" begin
        # Arrange
        abstract type DummyRule <: StrategyInterface.AbstractStrategy end

        # Act / Assert
        @test_throws MethodError StrategyInterface.context_kind(DummyRule)
        @test_throws MethodError StrategyInterface.internal_state_class(DummyRule)
    end

    @testset "sample_action convenience dispatch for no-context strategy" begin
        # Arrange
        strategy = LocalStrategies.DeterministicStrategy(:left)

        # Act
        result = StrategyInterface.sample_action(strategy)

        # Assert
        @test result == :left
    end

    @testset "sample_action convenience dispatch for contextual strategy" begin
        # Arrange
        strategy = LocalStrategies.observation_strategy((obs, rng) -> obs * 10)

        # Act
        result = StrategyInterface.sample_action(strategy, 3)

        # Assert
        @test result == 30
    end

    @testset "expected_value computes aligned weighted sum" begin
        # Arrange
        strategy = LocalStrategies.FiniteMixedStrategy((:a, :b), (0.25, 0.75))
        values = (2.0, 10.0)

        # Act
        result = StrategyInterface.expected_value(strategy, values)

        # Assert
        @test result ≈ 8.0
    end

    @testset "expected_value rejects misaligned values" begin
        # Arrange
        strategy = LocalStrategies.FiniteMixedStrategy((:a, :b), (0.5, 0.5))

        # Act / Assert
        @test_throws ArgumentError StrategyInterface.expected_value(strategy, (1.0,))
    end

    @testset "monte_carlo_expectation estimates deterministic strategy exactly" begin
        # Arrange
        rng = MersenneTwister(123)
        strategy = LocalStrategies.DeterministicStrategy(7)

        # Act
        result = StrategyInterface.monte_carlo_expectation(x -> 2x, strategy; rng=rng, n_samples=64)

        # Assert
        @test result == 14.0
    end

    @testset "monte_carlo_expectation rejects nonpositive sample count" begin
        # Arrange
        strategy = LocalStrategies.DeterministicStrategy(1)

        # Act / Assert
        @test_throws ArgumentError StrategyInterface.monte_carlo_expectation(identity, strategy; n_samples=0)
        @test_throws ArgumentError StrategyInterface.monte_carlo_expectation(identity, strategy; n_samples=-5)
    end
end