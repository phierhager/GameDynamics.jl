using Test

using GameLab.NormalForm
using GameLab.Stackelberg
using GameLab.DirectDecisionRules
using GameLab.Classification

@testset "Stackelberg" begin
    @testset "constructor and role accessors" begin
        p1 = [4.0 0.0; 3.0 2.0]
        p2 = [1.0 2.0; 0.0 3.0]
        base = NormalForm.NormalFormGame((p1, p2))

        g = Stackelberg.StackelbergGame(base, 1, 2)
        @test Stackelberg.leader(g) == 1
        @test Stackelberg.follower(g) == 2
        @test Classification.is_stackelberg_game(g)
        @test Classification.is_hierarchical_game(g)

        @test_throws ArgumentError Stackelberg.StackelbergGame(base, 0, 2)
        @test_throws ArgumentError Stackelberg.StackelbergGame(base, 1, 1)
    end

    @testset "follower_best_response arbitrary tie-break" begin
        p1 = [5.0 0.0; 1.0 4.0]
        p2 = [2.0 1.0; 0.0 3.0]
        base = NormalForm.NormalFormGame((p1, p2))
        g = Stackelberg.StackelbergGame(base, 1, 2)

        leader_strategy = DirectDecisionRules.FiniteMixedDecisionRule((1,), (1.0,))
        a_f, v_f = Stackelberg.follower_best_response(g, leader_strategy)

        @test a_f == 1
        @test isapprox(v_f, 2.0)
    end

    @testset "leader_value under follower response" begin
        p1 = [5.0 0.0; 1.0 4.0]
        p2 = [2.0 1.0; 0.0 3.0]
        base = NormalForm.NormalFormGame((p1, p2))
        g = Stackelberg.StackelbergGame(base, 1, 2)

        leader_strategy = DirectDecisionRules.FiniteMixedDecisionRule((1,), (1.0,))
        v = Stackelberg.leader_value(g, leader_strategy)

        @test isapprox(v, 5.0)
    end

    @testset "favor leader tie-break selects best for leader among follower ties" begin
        # If leader plays action 1, follower is indifferent between 1 and 2:
        # follower payoff = 1 for both columns.
        # leader prefers follower choose 1 since 10 > 0.
        p1 = [10.0 0.0; 2.0 2.0]
        p2 = [1.0 1.0; 0.0 2.0]
        base = NormalForm.NormalFormGame((p1, p2))
        g = Stackelberg.StackelbergGame(base, 1, 2)

        leader_strategy = DirectDecisionRules.FiniteMixedDecisionRule((1,), (1.0,))
        a_f, _ = Stackelberg.follower_best_response(
            g,
            leader_strategy;
            tie_break=Stackelberg.FAVOR_LEADER_TIE_BREAK,
        )

        @test a_f == 1
        @test isapprox(Stackelberg.leader_value(
            g,
            leader_strategy;
            tie_break=Stackelberg.FAVOR_LEADER_TIE_BREAK,
        ), 10.0)
    end

    @testset "invalid tie-break and illegal strategy domain" begin
        p1 = [1.0 0.0; 0.0 1.0]
        p2 = [1.0 0.0; 0.0 1.0]
        base = NormalForm.NormalFormGame((p1, p2))
        g = Stackelberg.StackelbergGame(base, 1, 2)

        bad_strategy = DirectDecisionRules.FiniteMixedDecisionRule((3,), (1.0,))
        @test_throws ArgumentError Stackelberg.follower_best_response(g, bad_strategy)
        @test_throws ArgumentError Stackelberg.leader_value(g, bad_strategy)

        good_strategy = DirectDecisionRules.FiniteMixedDecisionRule((1,), (1.0,))
        @test_throws ArgumentError Stackelberg.follower_best_response(
            g,
            good_strategy;
            tie_break=:not_a_policy,
        )
    end
end