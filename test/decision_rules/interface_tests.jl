using Test
using Random

using GameLab
using GameLab.DecisionRulesInterface
using GameLab.DirectDecisionRules

@testset "DecisionRulesInterface" begin
    @testset "trait defaults throw when not implemented" begin
        # Arrange
        abstract type DummyRule <: DecisionRulesInterface.AbstractDecisionRule end

        # Act / Assert
        @test_throws MethodError DecisionRulesInterface.context_kind(DummyRule)
        @test_throws MethodError DecisionRulesInterface.internal_state_class(DummyRule)
    end

    @testset "sample_action convenience dispatch for no-context rule" begin
        # Arrange
        rule = DirectDecisionRules.DeterministicDecisionRule(:left)

        # Act
        result = DecisionRulesInterface.sample_action(rule)

        # Assert
        @test result == :left
    end

    @testset "sample_action convenience dispatch for contextual rule" begin
        # Arrange
        rule = DirectDecisionRules.observation_rule((obs, rng) -> obs * 10)

        # Act
        result = DecisionRulesInterface.sample_action(rule, 3)

        # Assert
        @test result == 30
    end

    @testset "expected_value computes aligned weighted sum" begin
        # Arrange
        rule = DirectDecisionRules.FiniteMixedDecisionRule((:a, :b), (0.25, 0.75))
        values = (2.0, 10.0)

        # Act
        result = DecisionRulesInterface.expected_value(rule, values)

        # Assert
        @test result ≈ 8.0
    end

    @testset "expected_value rejects misaligned values" begin
        # Arrange
        rule = DirectDecisionRules.FiniteMixedDecisionRule((:a, :b), (0.5, 0.5))

        # Act / Assert
        @test_throws ArgumentError DecisionRulesInterface.expected_value(rule, (1.0,))
    end

    @testset "monte_carlo_expectation estimates deterministic rule exactly" begin
        # Arrange
        rng = MersenneTwister(123)
        rule = DirectDecisionRules.DeterministicDecisionRule(7)

        # Act
        result = DecisionRulesInterface.monte_carlo_expectation(x -> 2x, rule; rng=rng, n_samples=64)

        # Assert
        @test result == 14.0
    end

    @testset "monte_carlo_expectation rejects nonpositive sample count" begin
        # Arrange
        rule = DirectDecisionRules.DeterministicDecisionRule(1)

        # Act / Assert
        @test_throws ArgumentError DecisionRulesInterface.monte_carlo_expectation(identity, rule; n_samples=0)
        @test_throws ArgumentError DecisionRulesInterface.monte_carlo_expectation(identity, rule; n_samples=-5)
    end
end