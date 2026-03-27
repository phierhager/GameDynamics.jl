using Test
using Random

using GameLab
using GameLab.DecisionRulesInterface
using GameLab.JointDecisionRules

@testset "JointDecisionRules" begin
    @testset "CorrelatedActionRule canonicalizes duplicate tuples and normalizes" begin
        # Arrange
        support = ((:L, :U), (:R, :D), (:L, :U))
        probs = (1, 3, 2)

        # Act
        rule = JointDecisionRules.CorrelatedActionRule(support, probs)

        # Assert
        @test DecisionRulesInterface.context_kind(rule) isa DecisionRulesInterface.NoContext
        @test DecisionRulesInterface.internal_state_class(rule) isa DecisionRulesInterface.Stateless
        @test DecisionRulesInterface.support(rule) == ((:L, :U), (:R, :D))
        @test DecisionRulesInterface.probabilities(rule)[1] ≈ 0.5
        @test DecisionRulesInterface.probabilities(rule)[2] ≈ 0.5
    end

    @testset "sampling is reproducible with seeded RNG" begin
        # Arrange
        rng1 = MersenneTwister(99)
        rng2 = MersenneTwister(99)
        rule = JointDecisionRules.CorrelatedActionRule(
            ((:L, :U), (:R, :D), (:L, :D)),
            (0.2, 0.3, 0.5)
        )

        # Act
        draws1 = [DecisionRulesInterface.sample_action(rule, rng1) for _ in 1:20]
        draws2 = [DecisionRulesInterface.sample_action(rule, rng2) for _ in 1:20]

        # Assert
        @test draws1 == draws2
    end

    @testset "action_probability returns tuple mass or zero" begin
        # Arrange
        rule = JointDecisionRules.CorrelatedActionRule(
            ((:L, :U), (:R, :D)),
            (0.4, 0.6)
        )

        # Act
        p_hit = DecisionRulesInterface.action_probability(rule, (:R, :D))
        p_miss = DecisionRulesInterface.action_probability(rule, (:L, :D))

        # Assert
        @test p_hit ≈ 0.6
        @test p_miss == 0.0
    end

    @testset "constructor rejects invalid inputs" begin
        # Arrange / Act / Assert
        @test_throws ArgumentError JointDecisionRules.CorrelatedActionRule((), ())
        @test_throws ArgumentError JointDecisionRules.CorrelatedActionRule(((:L, :U),), (0.2, 0.8))
        @test_throws ArgumentError JointDecisionRules.CorrelatedActionRule(((:L, :U),), (-1.0,))
    end
end