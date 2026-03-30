using Test
using Random

using GameLab
using GameLab.StrategyInterface
using GameLab.LocalStrategies
using GameLab.StrategyProfiles
using .TestHelpers

@testset "StrategyProfiles" begin
    @testset "StrategyProfile construction and indexing" begin
        # Arrange
        strategies = (
            LocalStrategies.DeterministicStrategy(:a),
            LocalStrategies.DeterministicStrategy(:b)
        )

        # Act
        profile = StrategyProfiles.StrategyProfile(strategies)

        # Assert
        @test length(profile) == 2
        @test StrategyProfiles.num_strategies(profile) == 2
        @test profile[1].action == :a
        @test Tuple(profile) == strategies
        @test collect(profile) == collect(strategies)
        @test firstindex(profile) == 1
        @test lastindex(profile) == 2
    end

    @testset "StrategyProfile rejects non-strategy entries" begin
        # Arrange / Act / Assert
        @test_throws ArgumentError StrategyProfiles.StrategyProfile((
            LocalStrategies.DeterministicStrategy(:a),
            123
        ))
    end

    @testset "context_kinds and internal_state_classes introspect tuple of strategies" begin
        # Arrange
        profile = StrategyProfiles.StrategyProfile((
            LocalStrategies.DeterministicStrategy(:a),
            LocalStrategies.observation_strategy((obs, rng) -> obs; internal_state=StrategyInterface.Stateful())
        ))

        # Act
        cks = StrategyProfiles.context_kinds(profile)
        iscs = StrategyProfiles.internal_state_classes(profile)

        # Assert
        @test cks[1] isa StrategyInterface.NoContext
        @test cks[2] isa StrategyInterface.ObservationContext
        @test iscs[1] isa StrategyInterface.Stateless
        @test iscs[2] isa StrategyInterface.Stateful
    end

    @testset "context-homogeneity predicates" begin
        # Arrange
        uncond = StrategyProfiles.StrategyProfile((
            LocalStrategies.DeterministicStrategy(:a),
            LocalStrategies.FiniteMixedStrategy((:b, :c), (0.5, 0.5))
        ))

        obs = StrategyProfiles.StrategyProfile((
            LocalStrategies.observation_strategy((x, rng) -> x),
            LocalStrategies.observation_strategy((x, rng) -> x + 1)
        ))

        mixed = StrategyProfiles.StrategyProfile((
            LocalStrategies.DeterministicStrategy(:a),
            LocalStrategies.observation_strategy((x, rng) -> x)
        ))

        # Act / Assert
        @test StrategyProfiles.is_unconditioned(uncond)
        @test !StrategyProfiles.is_observation_conditioned(uncond)

        @test StrategyProfiles.is_observation_conditioned(obs)
        @test !StrategyProfiles.is_unconditioned(obs)

        @test !StrategyProfiles.is_unconditioned(mixed)
        @test !StrategyProfiles.is_observation_conditioned(mixed)
    end

    @testset "is_stateless distinguishes metadata correctly" begin
        # Arrange
        stateless_profile = StrategyProfiles.StrategyProfile((
            LocalStrategies.DeterministicStrategy(:a),
            LocalStrategies.DeterministicStrategy(:b)
        ))

        non_stateless_profile = StrategyProfiles.StrategyProfile((
            LocalStrategies.DeterministicStrategy(:a),
            TestHelpers.DummyStatefulRule(:b, StrategyInterface.Stateful())
        ))

        finite_state_profile = StrategyProfiles.StrategyProfile((
            LocalStrategies.DeterministicStrategy(:a),
            TestHelpers.DummyFiniteStateRule(:b)
        ))

        # Act / Assert
        @test StrategyProfiles.is_stateless(stateless_profile)
        @test !StrategyProfiles.is_stateless(non_stateless_profile)
        @test !StrategyProfiles.is_stateless(finite_state_profile)
    end

    @testset "require_* helpers return profile when valid and throw when invalid" begin
        # Arrange
        uncond = StrategyProfiles.StrategyProfile((
            LocalStrategies.DeterministicStrategy(:a),
            LocalStrategies.DeterministicStrategy(:b)
        ))

        obs = StrategyProfiles.StrategyProfile((
            LocalStrategies.observation_strategy((x, rng) -> x),
            LocalStrategies.observation_strategy((x, rng) -> x + 1)
        ))

        mixed = StrategyProfiles.StrategyProfile((
            LocalStrategies.DeterministicStrategy(:a),
            LocalStrategies.observation_strategy((x, rng) -> x)
        ))

        # Act / Assert
        @test StrategyProfiles.require_unconditioned(uncond) === uncond
        @test StrategyProfiles.require_observation_conditioned(obs) === obs
        @test_throws ArgumentError StrategyProfiles.require_unconditioned(mixed)
        @test_throws ArgumentError StrategyProfiles.require_observation_conditioned(mixed)
    end

    @testset "convenience profile constructors enforce homogeneity" begin
        # Arrange
        uncond_rules = (
            LocalStrategies.DeterministicStrategy(:a),
            LocalStrategies.DeterministicStrategy(:b)
        )

        obs_rules = (
            LocalStrategies.observation_strategy((x, rng) -> x),
            LocalStrategies.observation_strategy((x, rng) -> x + 1)
        )

        mixed_rules = (
            LocalStrategies.DeterministicStrategy(:a),
            LocalStrategies.observation_strategy((x, rng) -> x)
        )

        # Act / Assert
        @test StrategyProfiles.unconditioned_strategy_profile(uncond_rules) isa
              StrategyProfiles.StrategyProfile

        @test StrategyProfiles.observation_strategy_profile(obs_rules) isa
              StrategyProfiles.StrategyProfile

        @test_throws ArgumentError StrategyProfiles.unconditioned_strategy_profile(mixed_rules)
    end

    @testset "sample_joint_action works for unconditioned profiles only" begin
        # Arrange
        rng1 = MersenneTwister(42)
        rng2 = MersenneTwister(42)

        profile = StrategyProfiles.StrategyProfile((
            LocalStrategies.DeterministicStrategy(:L),
            LocalStrategies.FiniteMixedStrategy((:U, :D), (0.7, 0.3))
        ))

        mixed_profile = StrategyProfiles.StrategyProfile((
            LocalStrategies.DeterministicStrategy(:L),
            LocalStrategies.observation_strategy((x, rng) -> x)
        ))

        # Act
        draw1 = StrategyProfiles.sample_joint_action(profile, rng1)
        draw2 = StrategyProfiles.sample_joint_action(profile, rng2)

        # Assert
        @test draw1 == draw2
        @test draw1[1] == :L
        @test_throws ArgumentError StrategyProfiles.sample_joint_action(mixed_profile, MersenneTwister(1))
    end

    @testset "sample_joint_action tuple overloads work" begin
        # Arrange
        strategies = (
            LocalStrategies.DeterministicStrategy(:a),
            LocalStrategies.DeterministicStrategy(:b)
        )

        # Act
        draw = StrategyProfiles.sample_joint_action(strategies, MersenneTwister(1))

        # Assert
        @test draw == (:a, :b)
    end

    @testset "joint_action_probability multiplies local probabilities" begin
        # Arrange
        profile = StrategyProfiles.StrategyProfile((
            LocalStrategies.FiniteMixedStrategy((:a, :b), (0.25, 0.75)),
            LocalStrategies.FiniteMixedStrategy((:x, :y), (0.1, 0.9))
        ))

        # Act
        p = StrategyProfiles.joint_action_probability(profile, (:b, :y))

        # Assert
        @test p ≈ 0.75 * 0.9
    end

    @testset "joint_action_probability rejects mismatched tuple length" begin
        # Arrange
        profile = StrategyProfiles.StrategyProfile((
            LocalStrategies.DeterministicStrategy(:a),
            LocalStrategies.DeterministicStrategy(:b)
        ))

        # Act / Assert
        @test_throws ArgumentError StrategyProfiles.joint_action_probability(profile, (:a,))
    end

    @testset "joint_action_density multiplies local densities" begin
        # Arrange
        profile = StrategyProfiles.StrategyProfile((
            LocalStrategies.SamplerDensityStrategy(r -> 0.1, x -> x + 1, 0:1),
            LocalStrategies.SamplerDensityStrategy(r -> 0.2, x -> 2x, 0:1)
        ))

        # Act
        d = StrategyProfiles.joint_action_density(profile, (2.0, 3.0))

        # Assert
        @test d == (2.0 + 1) * (2 * 3.0)
    end

    @testset "joint_action_density rejects mismatched tuple length" begin
        # Arrange
        profile = StrategyProfiles.StrategyProfile((
            LocalStrategies.SamplerDensityStrategy(r -> 0.1, x -> x, 0:1),
            LocalStrategies.SamplerDensityStrategy(r -> 0.2, x -> x, 0:1)
        ))

        # Act / Assert
        @test_throws ArgumentError StrategyProfiles.joint_action_density(profile, (1.0,))
    end
end