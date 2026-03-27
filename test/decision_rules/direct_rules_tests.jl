using Test
using Random

using GameLab
using GameLab.DecisionRulesInterface
using GameLab.DirectDecisionRules
using .TestHelpers

@testset "DirectDecisionRules" begin
    @testset "DeterministicDecisionRule" begin
        @testset "traits and support metadata" begin
            # Arrange
            rule = DirectDecisionRules.DeterministicDecisionRule(:go)

            # Act / Assert
            @test DecisionRulesInterface.context_kind(rule) isa DecisionRulesInterface.NoContext
            @test DecisionRulesInterface.internal_state_class(rule) isa DecisionRulesInterface.Stateless
            @test DecisionRulesInterface.support(rule) == (:go,)
            @test DecisionRulesInterface.probabilities(rule) == (1.0,)
        end

        @testset "sampling and action_probability" begin
            # Arrange
            rng = MersenneTwister(1)
            rule = DirectDecisionRules.DeterministicDecisionRule(:go)

            # Act
            sampled = DecisionRulesInterface.sample_action(rule, rng)
            p_hit = DecisionRulesInterface.action_probability(rule, :go)
            p_miss = DecisionRulesInterface.action_probability(rule, :stop)

            # Assert
            @test sampled == :go
            @test p_hit == 1.0
            @test p_miss == 0.0
        end
    end

    @testset "FiniteMixedDecisionRule" begin
        @testset "canonicalizes duplicate actions and normalizes probabilities" begin
            # Arrange
            actions = [:a, :b, :a]
            probs = (1, 1, 2)

            # Act
            rule = DirectDecisionRules.FiniteMixedDecisionRule(actions, probs)

            # Assert
            @test DecisionRulesInterface.support(rule) == (:a, :b)
            @test DecisionRulesInterface.probabilities(rule)[1] ≈ 0.75
            @test DecisionRulesInterface.probabilities(rule)[2] ≈ 0.25
        end

        @testset "index-based convenience constructor" begin
            # Arrange
            probs = (2, 3, 5)

            # Act
            rule = DirectDecisionRules.FiniteMixedDecisionRule(probs)

            # Assert
            @test DecisionRulesInterface.support(rule) == (1, 2, 3)
            @test DecisionRulesInterface.probabilities(rule) == (0.2, 0.3, 0.5)
        end

        @testset "action_probability returns exact mass for present and zero for absent" begin
            # Arrange
            rule = DirectDecisionRules.FiniteMixedDecisionRule((:x, :y), (0.4, 0.6))

            # Act
            px = DecisionRulesInterface.action_probability(rule, :x)
            py = DecisionRulesInterface.action_probability(rule, :y)
            pz = DecisionRulesInterface.action_probability(rule, :z)

            # Assert
            @test px ≈ 0.4
            @test py ≈ 0.6
            @test pz == 0.0
        end

        @testset "sampling is reproducible with seeded RNG" begin
            # Arrange
            rng1 = MersenneTwister(55)
            rng2 = MersenneTwister(55)
            rule = DirectDecisionRules.FiniteMixedDecisionRule((:a, :b, :c), (0.2, 0.3, 0.5))

            # Act
            draws1 = [DecisionRulesInterface.sample_action(rule, rng1) for _ in 1:20]
            draws2 = [DecisionRulesInterface.sample_action(rule, rng2) for _ in 1:20]

            # Assert
            @test draws1 == draws2
        end

        @testset "constructor rejects invalid supports or probabilities" begin
            # Arrange / Act / Assert
            @test_throws ArgumentError DirectDecisionRules.FiniteMixedDecisionRule((), ())
            @test_throws ArgumentError DirectDecisionRules.FiniteMixedDecisionRule((:a,), (0.2, 0.8))
            @test_throws ArgumentError DirectDecisionRules.FiniteMixedDecisionRule((:a, :b), (-1.0, 2.0))
        end
    end

    @testset "ContextualDecisionRule" begin
        @testset "no-context sampling uses sampler(rng)" begin
            # Arrange
            rng = MersenneTwister(9)
            seen = Ref(false)
            rule = DirectDecisionRules.unconditioned_rule(function (r)
                seen[] = true
                return rand(r) < 1.0 ? :ok : :bad
            end; likelihood = a -> a == :ok ? 1.0 : 0.0)

            # Act
            sampled = DecisionRulesInterface.sample_action(rule, rng)
            prob = DecisionRulesInterface.action_probability(rule, :ok)

            # Assert
            @test seen[]
            @test sampled == :ok
            @test prob == 1.0
        end

        @testset "contextual sampling and probability use context-aware call signatures" begin
            # Arrange
            rule = DirectDecisionRules.observation_rule(
                (obs, rng) -> obs + 1;
                likelihood = (obs, action) -> action == obs + 1 ? 1.0 : 0.0
            )

            # Act
            sampled = DecisionRulesInterface.sample_action(rule, 10, MersenneTwister(1))
            p_hit = DecisionRulesInterface.action_probability(rule, 10, 11)
            p_miss = DecisionRulesInterface.action_probability(rule, 10, 9)

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
            @test_throws MethodError DecisionRulesInterface.action_probability(noctx, :a)
            @test_throws MethodError DecisionRulesInterface.action_probability(contextual, :obs, :a)
        end

        @testset "convenience constructors set correct context kinds" begin
            # Arrange / Act / Assert
            @test DecisionRulesInterface.context_kind(
                DirectDecisionRules.unconditioned_rule(rng -> 1)
            ) isa DecisionRulesInterface.NoContext

            @test DecisionRulesInterface.context_kind(
                DirectDecisionRules.observation_rule((c, rng) -> c)
            ) isa DecisionRulesInterface.ObservationContext

            @test DecisionRulesInterface.context_kind(
                DirectDecisionRules.state_rule((c, rng) -> c)
            ) isa DecisionRulesInterface.StateContext

            @test DecisionRulesInterface.context_kind(
                DirectDecisionRules.history_rule((c, rng) -> c)
            ) isa DecisionRulesInterface.HistoryContext

            @test DecisionRulesInterface.context_kind(
                DirectDecisionRules.infoset_rule((c, rng) -> c)
            ) isa DecisionRulesInterface.InfosetContext

            @test DecisionRulesInterface.context_kind(
                DirectDecisionRules.custom_context_rule((c, rng) -> c)
            ) isa DecisionRulesInterface.CustomContext
        end

        @testset "constructor preserves explicit internal state class" begin
            # Arrange
            rule = DirectDecisionRules.observation_rule(
                (obs, rng) -> obs;
                internal_state = DecisionRulesInterface.Stateful()
            )

            # Act / Assert
            @test DecisionRulesInterface.internal_state_class(rule) isa DecisionRulesInterface.Stateful
        end
    end

    @testset "SamplerDecisionRule and SamplerDensityDecisionRule" begin
        @testset "sampler rule exposes domain and samples from sampler" begin
            # Arrange
            rng = MersenneTwister(3)
            rule = DirectDecisionRules.SamplerDecisionRule(r -> rand(r), 0.0:0.1:1.0)

            # Act
            val = DecisionRulesInterface.sample_action(rule, rng)
            dom = DecisionRulesInterface.support(rule)

            # Assert
            @test 0.0 <= val <= 1.0
            @test dom == 0.0:0.1:1.0
            @test DecisionRulesInterface.context_kind(rule) isa DecisionRulesInterface.NoContext
        end

        @testset "sampler density rule evaluates provided density function" begin
            # Arrange
            rule = DirectDecisionRules.SamplerDensityDecisionRule(r -> 0.25, x -> 2x, 0.0:0.5:1.0)

            # Act
            val = DecisionRulesInterface.sample_action(rule, MersenneTwister(1))
            density = DecisionRulesInterface.action_density(rule, 0.75)

            # Assert
            @test val == 0.25
            @test density == 1.5
            @test DecisionRulesInterface.support(rule) == 0.0:0.5:1.0
        end
    end
end