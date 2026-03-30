using Test
using Random

using GameLab
using GameLab.StrategyInterface
using GameLab.JointStrategies

@testset "JointStrategies" begin
    @testset "CorrelatedRecommendationDevice canonicalizes duplicate tuples and normalizes" begin
        # Arrange
        support = ((:L, :U), (:R, :D), (:L, :U))
        probs = (1, 3, 2)

        # Act
        strategy = JointStrategies.CorrelatedRecommendationDevice(support, probs)

        # Assert
        @test StrategyInterface.context_kind(strategy) isa StrategyInterface.NoContext
        @test StrategyInterface.internal_state_class(strategy) isa StrategyInterface.Stateless
        @test StrategyInterface.support(strategy) == ((:L, :U), (:R, :D))
        @test StrategyInterface.probabilities(strategy)[1] ≈ 0.5
        @test StrategyInterface.probabilities(strategy)[2] ≈ 0.5
    end

    @testset "sampling is reproducible with seeded RNG" begin
        # Arrange
        rng1 = MersenneTwister(99)
        rng2 = MersenneTwister(99)
        strategy = JointStrategies.CorrelatedRecommendationDevice(
            ((:L, :U), (:R, :D), (:L, :D)),
            (0.2, 0.3, 0.5)
        )

        # Act
        draws1 = [StrategyInterface.sample_action(strategy, rng1) for _ in 1:20]
        draws2 = [StrategyInterface.sample_action(strategy, rng2) for _ in 1:20]

        # Assert
        @test draws1 == draws2
    end

    @testset "action_probability returns tuple mass or zero" begin
        # Arrange
        strategy = JointStrategies.CorrelatedRecommendationDevice(
            ((:L, :U), (:R, :D)),
            (0.4, 0.6)
        )

        # Act
        p_hit = StrategyInterface.action_probability(strategy, (:R, :D))
        p_miss = StrategyInterface.action_probability(strategy, (:L, :D))

        # Assert
        @test p_hit ≈ 0.6
        @test p_miss == 0.0
    end

    @testset "constructor rejects invalid inputs" begin
        # Arrange / Act / Assert
        @test_throws ArgumentError JointStrategies.CorrelatedRecommendationDevice((), ())
        @test_throws ArgumentError JointStrategies.CorrelatedRecommendationDevice(((:L, :U),), (0.2, 0.8))
        @test_throws ArgumentError JointStrategies.CorrelatedRecommendationDevice(((:L, :U),), (-1.0,))
    end
end