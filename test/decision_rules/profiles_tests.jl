using Test
using Random

using GameLab
using GameLab.DecisionRulesInterface
using GameLab.DirectDecisionRules
using GameLab.DecisionRuleProfiles
using .TestHelpers

@testset "DecisionRuleProfiles" begin
    @testset "DecisionRuleProfile construction and indexing" begin
        # Arrange
        rules = (
            DirectDecisionRules.DeterministicDecisionRule(:a),
            DirectDecisionRules.DeterministicDecisionRule(:b)
        )

        # Act
        profile = DecisionRuleProfiles.DecisionRuleProfile(rules)

        # Assert
        @test length(profile) == 2
        @test DecisionRuleProfiles.num_rules(profile) == 2
        @test profile[1].action == :a
        @test Tuple(profile) == rules
        @test collect(profile) == collect(rules)
        @test firstindex(profile) == 1
        @test lastindex(profile) == 2
    end

    @testset "DecisionRuleProfile rejects non-rule entries" begin
        # Arrange / Act / Assert
        @test_throws ArgumentError DecisionRuleProfiles.DecisionRuleProfile((
            DirectDecisionRules.DeterministicDecisionRule(:a),
            123
        ))
    end

    @testset "context_kinds and internal_state_classes introspect tuple of rules" begin
        # Arrange
        profile = DecisionRuleProfiles.DecisionRuleProfile((
            DirectDecisionRules.DeterministicDecisionRule(:a),
            DirectDecisionRules.observation_rule((obs, rng) -> obs; internal_state=DecisionRulesInterface.Stateful())
        ))

        # Act
        cks = DecisionRuleProfiles.context_kinds(profile)
        iscs = DecisionRuleProfiles.internal_state_classes(profile)

        # Assert
        @test cks[1] isa DecisionRulesInterface.NoContext
        @test cks[2] isa DecisionRulesInterface.ObservationContext
        @test iscs[1] isa DecisionRulesInterface.Stateless
        @test iscs[2] isa DecisionRulesInterface.Stateful
    end

    @testset "context-homogeneity predicates" begin
        # Arrange
        uncond = DecisionRuleProfiles.DecisionRuleProfile((
            DirectDecisionRules.DeterministicDecisionRule(:a),
            DirectDecisionRules.FiniteMixedDecisionRule((:b, :c), (0.5, 0.5))
        ))

        obs = DecisionRuleProfiles.DecisionRuleProfile((
            DirectDecisionRules.observation_rule((x, rng) -> x),
            DirectDecisionRules.observation_rule((x, rng) -> x + 1)
        ))

        mixed = DecisionRuleProfiles.DecisionRuleProfile((
            DirectDecisionRules.DeterministicDecisionRule(:a),
            DirectDecisionRules.observation_rule((x, rng) -> x)
        ))

        # Act / Assert
        @test DecisionRuleProfiles.is_unconditioned(uncond)
        @test !DecisionRuleProfiles.is_observation_conditioned(uncond)

        @test DecisionRuleProfiles.is_observation_conditioned(obs)
        @test !DecisionRuleProfiles.is_unconditioned(obs)

        @test !DecisionRuleProfiles.is_unconditioned(mixed)
        @test !DecisionRuleProfiles.is_observation_conditioned(mixed)
    end

    @testset "is_stateless distinguishes metadata correctly" begin
        # Arrange
        stateless_profile = DecisionRuleProfiles.DecisionRuleProfile((
            DirectDecisionRules.DeterministicDecisionRule(:a),
            DirectDecisionRules.DeterministicDecisionRule(:b)
        ))

        non_stateless_profile = DecisionRuleProfiles.DecisionRuleProfile((
            DirectDecisionRules.DeterministicDecisionRule(:a),
            TestHelpers.DummyStatefulRule(:b, DecisionRulesInterface.Stateful())
        ))

        finite_state_profile = DecisionRuleProfiles.DecisionRuleProfile((
            DirectDecisionRules.DeterministicDecisionRule(:a),
            TestHelpers.DummyFiniteStateRule(:b)
        ))

        # Act / Assert
        @test DecisionRuleProfiles.is_stateless(stateless_profile)
        @test !DecisionRuleProfiles.is_stateless(non_stateless_profile)
        @test !DecisionRuleProfiles.is_stateless(finite_state_profile)
    end

    @testset "require_* helpers return profile when valid and throw when invalid" begin
        # Arrange
        uncond = DecisionRuleProfiles.DecisionRuleProfile((
            DirectDecisionRules.DeterministicDecisionRule(:a),
            DirectDecisionRules.DeterministicDecisionRule(:b)
        ))

        obs = DecisionRuleProfiles.DecisionRuleProfile((
            DirectDecisionRules.observation_rule((x, rng) -> x),
            DirectDecisionRules.observation_rule((x, rng) -> x + 1)
        ))

        mixed = DecisionRuleProfiles.DecisionRuleProfile((
            DirectDecisionRules.DeterministicDecisionRule(:a),
            DirectDecisionRules.observation_rule((x, rng) -> x)
        ))

        # Act / Assert
        @test DecisionRuleProfiles.require_unconditioned(uncond) === uncond
        @test DecisionRuleProfiles.require_observation_conditioned(obs) === obs
        @test_throws ArgumentError DecisionRuleProfiles.require_unconditioned(mixed)
        @test_throws ArgumentError DecisionRuleProfiles.require_observation_conditioned(mixed)
    end

    @testset "convenience profile constructors enforce homogeneity" begin
        # Arrange
        uncond_rules = (
            DirectDecisionRules.DeterministicDecisionRule(:a),
            DirectDecisionRules.DeterministicDecisionRule(:b)
        )

        obs_rules = (
            DirectDecisionRules.observation_rule((x, rng) -> x),
            DirectDecisionRules.observation_rule((x, rng) -> x + 1)
        )

        mixed_rules = (
            DirectDecisionRules.DeterministicDecisionRule(:a),
            DirectDecisionRules.observation_rule((x, rng) -> x)
        )

        # Act / Assert
        @test DecisionRuleProfiles.unconditioned_rule_profile(uncond_rules) isa
              DecisionRuleProfiles.DecisionRuleProfile

        @test DecisionRuleProfiles.observation_rule_profile(obs_rules) isa
              DecisionRuleProfiles.DecisionRuleProfile

        @test_throws ArgumentError DecisionRuleProfiles.unconditioned_rule_profile(mixed_rules)
    end

    @testset "sample_joint_action works for unconditioned profiles only" begin
        # Arrange
        rng1 = MersenneTwister(42)
        rng2 = MersenneTwister(42)

        profile = DecisionRuleProfiles.DecisionRuleProfile((
            DirectDecisionRules.DeterministicDecisionRule(:L),
            DirectDecisionRules.FiniteMixedDecisionRule((:U, :D), (0.7, 0.3))
        ))

        mixed_profile = DecisionRuleProfiles.DecisionRuleProfile((
            DirectDecisionRules.DeterministicDecisionRule(:L),
            DirectDecisionRules.observation_rule((x, rng) -> x)
        ))

        # Act
        draw1 = DecisionRuleProfiles.sample_joint_action(profile, rng1)
        draw2 = DecisionRuleProfiles.sample_joint_action(profile, rng2)

        # Assert
        @test draw1 == draw2
        @test draw1[1] == :L
        @test_throws ArgumentError DecisionRuleProfiles.sample_joint_action(mixed_profile, MersenneTwister(1))
    end

    @testset "sample_joint_action tuple overloads work" begin
        # Arrange
        rules = (
            DirectDecisionRules.DeterministicDecisionRule(:a),
            DirectDecisionRules.DeterministicDecisionRule(:b)
        )

        # Act
        draw = DecisionRuleProfiles.sample_joint_action(rules, MersenneTwister(1))

        # Assert
        @test draw == (:a, :b)
    end

    @testset "joint_probability multiplies local probabilities" begin
        # Arrange
        profile = DecisionRuleProfiles.DecisionRuleProfile((
            DirectDecisionRules.FiniteMixedDecisionRule((:a, :b), (0.25, 0.75)),
            DirectDecisionRules.FiniteMixedDecisionRule((:x, :y), (0.1, 0.9))
        ))

        # Act
        p = DecisionRuleProfiles.joint_probability(profile, (:b, :y))

        # Assert
        @test p ≈ 0.75 * 0.9
    end

    @testset "joint_probability rejects mismatched tuple length" begin
        # Arrange
        profile = DecisionRuleProfiles.DecisionRuleProfile((
            DirectDecisionRules.DeterministicDecisionRule(:a),
            DirectDecisionRules.DeterministicDecisionRule(:b)
        ))

        # Act / Assert
        @test_throws ArgumentError DecisionRuleProfiles.joint_probability(profile, (:a,))
    end

    @testset "joint_density multiplies local densities" begin
        # Arrange
        profile = DecisionRuleProfiles.DecisionRuleProfile((
            DirectDecisionRules.SamplerDensityDecisionRule(r -> 0.1, x -> x + 1, 0:1),
            DirectDecisionRules.SamplerDensityDecisionRule(r -> 0.2, x -> 2x, 0:1)
        ))

        # Act
        d = DecisionRuleProfiles.joint_density(profile, (2.0, 3.0))

        # Assert
        @test d == (2.0 + 1) * (2 * 3.0)
    end

    @testset "joint_density rejects mismatched tuple length" begin
        # Arrange
        profile = DecisionRuleProfiles.DecisionRuleProfile((
            DirectDecisionRules.SamplerDensityDecisionRule(r -> 0.1, x -> x, 0:1),
            DirectDecisionRules.SamplerDensityDecisionRule(r -> 0.2, x -> x, 0:1)
        ))

        # Act / Assert
        @test_throws ArgumentError DecisionRuleProfiles.joint_density(profile, (1.0,))
    end
end