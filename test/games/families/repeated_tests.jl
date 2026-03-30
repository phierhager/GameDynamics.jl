using Test
using Random

using GameLab.Kernel
using GameLab.Classification
using GameLab.NormalForm
using GameLab.RepeatedGames
using GameLab.LocalStrategies

@testset "RepeatedGames" begin
    p1 = [3.0 0.0; 5.0 1.0]
    p2 = [3.0 5.0; 0.0 1.0]
    stage = NormalForm.NormalFormGame((p1, p2))

    @testset "history objects" begin
        h = RepeatedGames.empty_history(stage)

        @test RepeatedGames.num_rounds(h) == 0
        @test RepeatedGames.current_round(h) == 1

        push!(h.rounds, RepeatedGames.RepeatedRoundRecord{2}((1, 2), (0.0, 5.0)))
        @test RepeatedGames.num_rounds(h) == 1
        @test RepeatedGames.current_round(h) == 2
        @test RepeatedGames.round_actions(h, 1) == (1, 2)
        @test RepeatedGames.round_payoffs(h, 1) == (0.0, 5.0)
    end

    @testset "repeated-game constructors" begin
        g = RepeatedGames.RepeatedNormalFormGame(stage; horizon=4, discount=0.9)
        @test g.horizon == 4
        @test g.discount == 0.9
        @test Classification.is_repeated_game(g)

        gg = RepeatedGames.GeneralRepeatedNormalFormGame(stage; horizon=3, discount=1.0)
        @test gg.horizon == 3
        @test gg.discount == 1.0
        @test Classification.is_repeated_game(gg)

        @test_throws ArgumentError RepeatedGames.RepeatedNormalFormGame(stage; horizon=0)
        @test_throws ArgumentError RepeatedGames.RepeatedNormalFormGame(stage; horizon=2, discount=0.0)
        @test_throws ArgumentError RepeatedGames.GeneralRepeatedNormalFormGame(stage; horizon=0)
        @test_throws ArgumentError RepeatedGames.GeneralRepeatedNormalFormGame(stage; horizon=2, discount=1.5)
    end

    @testset "return helpers" begin
        xs = [1.0, 2.0, 3.0]
        @test RepeatedGames.undiscounted_return(xs) == 6.0
        @test isapprox(RepeatedGames.discounted_return(xs, 0.5), 1.0 + 0.5 * 2.0 + 0.25 * 3.0)
    end

    @testset "play_repeated_profile with deterministic stationary profile" begin
        g = RepeatedGames.RepeatedNormalFormGame(stage; horizon=3, discount=0.5)

        s1 = LocalStrategies.FiniteMixedStrategy((2,), (1.0,))  # always action 2
        s2 = LocalStrategies.FiniteMixedStrategy((1,), (1.0,))  # always action 1
        rng = MersenneTwister(1)

        totals = RepeatedGames.play_repeated_profile(g, (s1, s2), rng)

        # stage payoff at (2,1) = (5,0), repeated 3 rounds
        @test totals == (5.0 + 0.5 * 5.0 + 0.25 * 5.0, 0.0)
    end

    @testset "play_repeated_profile rejects wrong profile length" begin
        g = RepeatedGames.RepeatedNormalFormGame(stage; horizon=2, discount=1.0)
        s1 = LocalStrategies.FiniteMixedStrategy((1,), (1.0,))
        @test_throws ArgumentError RepeatedGames.play_repeated_profile(g, (s1,))
    end

    struct GrimLikeStrategy <: StrategyInterface.AbstractStrategy
        cooperate_on_first::Int
        defect_after::Int
    end

    StrategyInterface.sample_action(s::GrimLikeStrategy, history, rng::AbstractRNG=Random.default_rng()) =
        RepeatedGames.num_rounds(history) == 0 ? s.cooperate_on_first : s.defect_after

    @testset "play_general_repeated_profile returns totals and realized history" begin
        g = RepeatedGames.GeneralRepeatedNormalFormGame(stage; horizon=3, discount=1.0)

        s1 = GrimLikeStrategy(1, 2)
        s2 = GrimLikeStrategy(1, 2)
        rng = MersenneTwister(2)

        totals, history = RepeatedGames.play_general_repeated_profile(g, (s1, s2), rng)

        @test totals isa NTuple{2,Float64}
        @test history isa RepeatedGames.RepeatedHistory
        @test RepeatedGames.num_rounds(history) == 3

        @test RepeatedGames.round_actions(history, 1) == (1, 1)
        @test RepeatedGames.round_actions(history, 2) == (2, 2)
        @test RepeatedGames.round_actions(history, 3) == (2, 2)
    end

    struct BadGeneralStrategy <: StrategyInterface.AbstractStrategy end
    StrategyInterface.sample_action(::BadGeneralStrategy, history, rng::AbstractRNG=Random.default_rng()) = 99

    @testset "play_general_repeated_profile rejects illegal actions" begin
        g = RepeatedGames.GeneralRepeatedNormalFormGame(stage; horizon=2, discount=1.0)
        @test_throws ArgumentError RepeatedGames.play_general_repeated_profile(
            g,
            (BadGeneralStrategy(), BadGeneralStrategy()),
            MersenneTwister(3),
        )
    end
end