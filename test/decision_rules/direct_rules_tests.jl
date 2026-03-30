using Test
using Random

using GameLab
using GameLab.StrategyInterface
using GameLab.LocalStrategies
using .TestHelpers

@testset "LocalStrategies" begin
    @testset "DeterministicStrategy" begin
        @testset "traits and support metadata" begin
            # Arrange
            strategy = LocalStrategies.DeterministicStrategy(:go)

            # Act / Assert
            @test StrategyInterface.context_kind(strategy) isa StrategyInterface.NoContext
            @test StrategyInterface.internal_state_class(strategy) isa StrategyInterface.Stateless
            @test StrategyInterface.support(strategy) == (:go,)
            @test StrategyInterface.probabilities(strategy) == (1.0,)
        end

        @testset "sampling and action_probability" begin
            # Arrange
            rng = MersenneTwister(1)
            strategy = LocalStrategies.DeterministicStrategy(:go)

            # Act
            sampled = StrategyInterface.sample_action(strategy, rng)
            p_hit = StrategyInterface.action_probability(strategy, :go)
            p_miss = StrategyInterface.action_probability(strategy, :stop)

            # Assert
            @test sampled == :go
            @test p_hit == 1.0
            @test p_miss == 0.0
        end
    end

    @testset "FiniteMixedStrategy" begin
        @testset "canonicalizes duplicate actions and normalizes probabilities" begin
            # Arrange
            actions = [:a, :b, :a]
            probs = (1, 1, 2)

            # Act
            strategy = LocalStrategies.FiniteMixedStrategy(actions, probs)

            # Assert
            @test StrategyInterface.support(strategy) == (:a, :b)
            @test StrategyInterface.probabilities(strategy)[1] ≈ 0.75
            @test StrategyInterface.probabilities(strategy)[2] ≈ 0.25
        end

        @testset "index-based convenience constructor" begin
            # Arrange
            probs = (2, 3, 5)

            # Act
            strategy = LocalStrategies.FiniteMixedStrategy(probs)

            # Assert
            @test StrategyInterface.support(strategy) == (1, 2, 3)
            @test StrategyInterface.probabilities(strategy) == (0.2, 0.3, 0.5)
        end

        @testset "action_probability returns exact mass for present and zero for absent" begin
            # Arrange
            strategy = LocalStrategies.FiniteMixedStrategy((:x, :y), (0.4, 0.6))

            # Act
            px = StrategyInterface.action_probability(strategy, :x)
            py = StrategyInterface.action_probability(strategy, :y)
            pz = StrategyInterface.action_probability(strategy, :z)

            # Assert
            @test px ≈ 0.4
            @test py ≈ 0.6
            @test pz == 0.0
        end

        @testset "sampling is reproducible with seeded RNG" begin
            # Arrange
            rng1 = MersenneTwister(55)
            rng2 = MersenneTwister(55)
            strategy = LocalStrategies.FiniteMixedStrategy((:a, :b, :c), (0.2, 0.3, 0.5))

            # Act
            draws1 = [StrategyInterface.sample_action(strategy, rng1) for _ in 1:20]
            draws2 = [StrategyInterface.sample_action(strategy, rng2) for _ in 1:20]

            # Assert
            @test draws1 == draws2
        end

        @testset "constructor rejects invalid supports or probabilities" begin
            # Arrange / Act / Assert
            @test_throws ArgumentError LocalStrategies.FiniteMixedStrategy((), ())
            @test_throws ArgumentError LocalStrategies.FiniteMixedStrategy((:a,), (0.2, 0.8))
            @test_throws ArgumentError LocalStrategies.FiniteMixedStrategy((:a, :b), (-1.0, 2.0))
        end
    end

    @testset "ContextualStrategy" begin
        @testset "no-context sampling uses sampler(rng)" begin
            # Arrange
            rng = MersenneTwister(9)
            seen = Ref(false)
            strategy = LocalStrategies.unconditioned_strategy(function (r)
                seen[] = true
                return rand(r) < 1.0 ? :ok : :bad
            end; likelihood = a -> a == :ok ? 1.0 : 0.0)

            # Act
            sampled = StrategyInterface.sample_action(strategy, rng)
            prob = StrategyInterface.action_probability(strategy, :ok)

            # Assert
            @test seen[]
            @test sampled == :ok
            @test prob == 1.0
        end

        @testset "contextual sampling and probability use context-aware call signatures" begin
            # Arrange
            strategy = LocalStrategies.observation_strategy(
                (obs, rng) -> obs + 1;
                likelihood = (obs, action) -> action == obs + 1 ? 1.0 : 0.0
            )

            # Act
            sampled = StrategyInterface.sample_action(strategy, 10, MersenneTwister(1))
            p_hit = StrategyInterface.action_probability(strategy, 10, 11)
            p_miss = StrategyInterface.action_probability(strategy, 10, 9)

            # Assert
            @test sampled == 11
            @test p_hit == 1.0
            @test p_miss == 0.0
        end

        @testset "action_probability throws MethodError when likelihood is missing" begin
            # Arrange
            noctx = TestHelpers.NoLikelihoodNoContextRule
            contextual = TestHelpers.NoLikelihoodObservationRule

            # Act / Assert
            @test_throws MethodError StrategyInterface.action_probability(noctx, :a)
            @test_throws MethodError StrategyInterface.action_probability(contextual, :obs, :a)
        end

        @testset "convenience constructors set correct context kinds" begin
            # Arrange / Act / Assert
            @test StrategyInterface.context_kind(
                LocalStrategies.unconditioned_strategy(rng -> 1)
            ) isa StrategyInterface.NoContext

            @test StrategyInterface.context_kind(
                LocalStrategies.observation_strategy((c, rng) -> c)
            ) isa StrategyInterface.ObservationContext

            @test StrategyInterface.context_kind(
                LocalStrategies.state_strategy((c, rng) -> c)
            ) isa StrategyInterface.StateContext

            @test StrategyInterface.context_kind(
                LocalStrategies.history_strategy((c, rng) -> c)
            ) isa StrategyInterface.HistoryContext

            @test StrategyInterface.context_kind(
                LocalStrategies.infoset_strategy((c, rng) -> c)
            ) isa StrategyInterface.InfosetContext

            @test StrategyInterface.context_kind(
                LocalStrategies.custom_context_strategy((c, rng) -> c)
            ) isa StrategyInterface.CustomContext
        end

        @testset "constructor preserves explicit internal state class" begin
            # Arrange
            strategy = LocalStrategies.observation_strategy(
                (obs, rng) -> obs;
                internal_state = StrategyInterface.Stateful()
            )

            # Act / Assert
            @test StrategyInterface.internal_state_class(strategy) isa StrategyInterface.Stateful
        end
    end

    @testset "SamplerStrategy and SamplerDensityStrategy" begin
        @testset "sampler strategy exposes domain and samples from sampler" begin
            # Arrange
            rng = MersenneTwister(3)
            strategy = LocalStrategies.SamplerStrategy(r -> rand(r), 0.0:0.1:1.0)

            # Act
            val = StrategyInterface.sample_action(strategy, rng)
            dom = StrategyInterface.support(strategy)

            # Assert
            @test 0.0 <= val <= 1.0
            @test dom == 0.0:0.1:1.0
            @test StrategyInterface.context_kind(strategy) isa StrategyInterface.NoContext
        end

        @testset "sampler density strategy evaluates provided density function" begin
            # Arrange
            strategy = LocalStrategies.SamplerDensityStrategy(r -> 0.25, x -> 2x, 0.0:0.5:1.0)

            # Act
            val = StrategyInterface.sample_action(strategy, MersenneTwister(1))
            density = StrategyInterface.action_density(strategy, 0.75)

            # Assert
            @test val == 0.25
            @test density == 1.5
            @test StrategyInterface.support(strategy) == 0.0:0.5:1.0
        end
    end
end