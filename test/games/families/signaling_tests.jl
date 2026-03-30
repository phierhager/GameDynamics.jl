using Test
using Random

using GameLab.Spaces
using GameLab.LocalStrategies
using GameLab.BayesianPriors
using GameLab.Signaling
using GameLab.Classification

@testset "Signaling" begin
    @testset "SenderReceiverRoles" begin
        r = Signaling.SenderReceiverRoles(1, 2)
        @test Signaling.sender(r) == 1
        @test Signaling.receiver(r) == 2

        @test_throws ArgumentError Signaling.SenderReceiverRoles(0, 2)
        @test_throws ArgumentError Signaling.SenderReceiverRoles(1, 1)
    end

    @testset "SignalingProfile accessors and sampling" begin
        sender_strat = LocalStrategies.FiniteMixedStrategy((:m1, :m2), (1.0, 0.0))
        receiver_strat = LocalStrategies.FiniteMixedStrategy((:a1, :a2), (0.0, 1.0))
        prof = Signaling.SignalingProfile(sender_strat, receiver_strat)

        @test Signaling.sender_strategy(prof) === sender_strat
        @test Signaling.receiver_strategy(prof) === receiver_strat
        @test Classification.is_signaling_game(prof)

        rng = MersenneTwister(1)
        @test Signaling.sample_message(prof, :any_type, rng) == :m1
        @test Signaling.sample_receiver_action(prof, :m1, rng) == :a2
    end

    struct SenderBehavior <: StrategyInterface.AbstractStrategy end
    StrategyInterface.local_strategy(::SenderBehavior, ::Symbol) = LocalStrategies.FiniteMixedStrategy((:m1, :m2), (0.75, 0.25))
    StrategyInterface.local_strategy(::SenderBehavior, ::Int) = LocalStrategies.FiniteMixedStrategy((:m2, :m3), (0.2, 0.8))

    @testset "induced_message_distribution over finite sender type space" begin
        prior = BayesianPriors.CommonPrior(
            ((:t1, :x), (1, :y)),
            (0.4, 0.6),
        )

        d = Signaling.induced_message_distribution(prior, SenderBehavior(), 1)

        @test d isa LocalStrategies.FiniteMixedStrategy
        supp = StrategyInterface.support(d)
        probs = StrategyInterface.probabilities(d)

        @test supp == (:m1, :m2, :m3)
        @test isapprox(sum(probs), 1.0)
        @test isapprox(StrategyInterface.action_probability(d, :m1), 0.4 * 0.75)
        @test isapprox(StrategyInterface.action_probability(d, :m2), 0.4 * 0.25 + 0.6 * 0.2)
        @test isapprox(StrategyInterface.action_probability(d, :m3), 0.6 * 0.8)
    end

    @testset "induced_message_distribution rejects empty local support" begin
        struct BadSenderBehavior <: StrategyInterface.AbstractStrategy end
        StrategyInterface.local_strategy(::BadSenderBehavior, t) = LocalStrategies.FiniteMixedStrategy((), ())

        prior = BayesianPriors.CommonPrior(((:t1,),), (1.0,))
        @test_throws Exception Signaling.induced_message_distribution(prior, BadSenderBehavior(), 1)
    end

    @testset "induced_message_distribution validates sender player index" begin
        prior = BayesianPriors.CommonPrior(((:t1, :x),), (1.0,))
        @test_throws ArgumentError Signaling.induced_message_distribution(prior, SenderBehavior(), 0)
    end
end