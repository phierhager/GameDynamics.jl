using Test
using Random

using GameLab.Kernel
using GameLab.Spec
using GameLab.Classification
using GameLab.NormalForm
using GameLab.LocalStrategies

@testset "NormalForm" begin
    @testset "constructor and inferred metadata" begin
        p1 = [1.0 -1.0; -1.0 1.0]
        p2 = -p1
        g = NormalForm.NormalFormGame((p1, p2))

        @test Kernel.num_players(g) == 2
        @test NormalForm.action_count(g, 1) == 2
        @test NormalForm.action_count(g, 2) == 2
        @test NormalForm.action_counts(g) == (2, 2)

        spec = Spec.game_spec(g)
        @test spec.horizon_kind == Spec.EPISODIC
        @test spec.payoff_kind == Spec.ZERO_SUM
        @test spec.max_steps == 1
        @test spec.default_discount == 1.0
        @test spec.perfect_information === false
        @test spec.stochastic === false
        @test spec.simultaneous_moves === true
        @test spec.cooperative === false
        @test spec.reward_sharing == Spec.INDEPENDENT_REWARD

        @test Classification.is_normal_form(g)
        @test !Classification.is_extensive_form(g)
    end

    @testset "constant-sum and general-sum inference" begin
        p1 = [2.0 0.0; 1.0 3.0]
        p2 = [1.0 3.0; 2.0 0.0]   # sums to constant 3
        g1 = NormalForm.NormalFormGame((p1, p2))
        @test Spec.game_spec(g1).payoff_kind == Spec.CONSTANT_SUM

        q1 = [2.0 0.0; 1.0 3.0]
        q2 = [1.0 1.0; 2.0 0.0]   # not constant-sum
        g2 = NormalForm.NormalFormGame((q1, q2))
        @test Spec.game_spec(g2).payoff_kind == Spec.GENERAL_SUM
    end

    @testset "constructor rejects inconsistent tensors" begin
        @test_throws ArgumentError NormalForm.NormalFormGame(())
        @test_throws ArgumentError NormalForm.NormalFormGame((
            [1.0 2.0; 3.0 4.0],
            [1.0, 2.0],
        ))
        @test_throws ArgumentError NormalForm.NormalFormGame((
            ones(2, 2),
            ones(2, 3),
        ))
    end

    @testset "kernel integration" begin
        p1 = [3.0 0.0; 5.0 1.0]
        p2 = [3.0 5.0; 0.0 1.0]
        g = NormalForm.NormalFormGame((p1, p2))
        s = Kernel.init_state(g)

        @test Kernel.action_mode(typeof(g)) == Kernel.IndexedActions
        @test Kernel.has_action_mask(typeof(g)) == true
        @test Kernel.node_kind(g, s) == Kernel.SIMULTANEOUS
        @test Tuple(Kernel.active_players(g, s)) == (1, 2)

        @test Kernel.legal_actions(g, s, 1) == Base.OneTo(2)
        @test Kernel.legal_actions(g, s, 2) == Base.OneTo(2)
        @test Kernel.indexed_action_count(g, 1) == 2
        @test Kernel.legal_action_mask(g, s, 1) == (true, true)

        s2, rewards = Kernel.step(g, s, Kernel.joint_action(2, 1))
        @test s2 isa NormalForm.NormalFormState
        @test Kernel.node_kind(g, s2) == Kernel.TERMINAL
        @test rewards == (5.0, 0.0)

        @test_throws ArgumentError Kernel.step(g, s2, Kernel.joint_action(1, 1))
        @test_throws ArgumentError Kernel.step(g, s, Kernel.joint_action(3, 1))
    end

    @testset "pure_payoff and support_profiles" begin
        p1 = [1.0 2.0; 3.0 4.0]
        p2 = [4.0 3.0; 2.0 1.0]
        g = NormalForm.NormalFormGame((p1, p2))

        @test NormalForm.pure_payoff(g, (1, 2)) == (2.0, 3.0)

        profiles = collect(NormalForm.support_profiles(g))
        @test length(profiles) == 4
        @test (1, 1) in profiles
        @test (2, 2) in profiles
    end

    @testset "expected_payoff for mixed strategies (2-player)" begin
        p1 = [2.0 0.0; 1.0 3.0]
        p2 = [1.0 4.0; 0.0 2.0]
        g = NormalForm.NormalFormGame((p1, p2))

        s1 = LocalStrategies.FiniteMixedStrategy((1, 2), (0.25, 0.75))
        s2 = LocalStrategies.FiniteMixedStrategy((1, 2), (0.5, 0.5))

        v1, v2 = NormalForm.expected_payoff(g, (s1, s2))

        expected1 = 0.25 * (0.5 * 2.0 + 0.5 * 0.0) + 0.75 * (0.5 * 1.0 + 0.5 * 3.0)
        expected2 = 0.25 * (0.5 * 1.0 + 0.5 * 4.0) + 0.75 * (0.5 * 0.0 + 0.5 * 2.0)

        @test isapprox(v1, expected1)
        @test isapprox(v2, expected2)
    end

    @testset "expected_payoff for mixed strategies (N-player)" begin
        p1 = reshape(Float64[
            1, 2, 3, 4,
            5, 6, 7, 8
        ], 2, 2, 2)
        p2 = p1 .+ 10
        p3 = p1 .+ 20

        g = NormalForm.NormalFormGame((p1, p2, p3))

        s1 = LocalStrategies.FiniteMixedStrategy((1, 2), (0.5, 0.5))
        s2 = LocalStrategies.FiniteMixedStrategy((1, 2), (1.0, 0.0))
        s3 = LocalStrategies.FiniteMixedStrategy((1, 2), (0.25, 0.75))

        vals = NormalForm.expected_payoff(g, (s1, s2, s3))
        @test vals isa NTuple{3,Float64}
    end

    @testset "expected_payoff for correlated strategy" begin
        p1 = [4.0 0.0; 1.0 3.0]
        p2 = [4.0 1.0; 0.0 3.0]
        g = NormalForm.NormalFormGame((p1, p2))

        corr = JointStrategies.CorrelatedRecommendationDevice(((1, 1), (2, 2)), (0.25, 0.75))
        v1, v2 = NormalForm.expected_payoff(g, corr)

        @test isapprox(v1, 0.25 * 4.0 + 0.75 * 3.0)
        @test isapprox(v2, 0.25 * 4.0 + 0.75 * 3.0)
    end

    @testset "best_response_values and best_response" begin
        p1 = [3.0 0.0; 5.0 1.0]
        p2 = [3.0 5.0; 0.0 1.0]
        g = NormalForm.NormalFormGame((p1, p2))

        s1 = LocalStrategies.FiniteMixedStrategy((1, 2), (0.5, 0.5))
        s2 = LocalStrategies.FiniteMixedStrategy((1, 2), (0.25, 0.75))

        vals1 = NormalForm.best_response_values(g, 1, (s1, s2))
        vals2 = NormalForm.best_response_values(g, 2, (s1, s2))

        @test length(vals1) == 2
        @test length(vals2) == 2

        a1, v1 = NormalForm.best_response(g, 1, (s1, s2))
        @test a1 == argmax(vals1)
        @test v1 == vals1[a1]

        @test_throws ArgumentError NormalForm.best_response_values(g, 3, (s1, s2))
    end
end