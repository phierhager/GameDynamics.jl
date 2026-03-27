using Test
using GameLab.Spec
using GameLab.Kernel
using .TestMockGames

@testset "Spec" begin
    @testset "enum values construct and compare" begin
        @test Spec.EPISODIC isa Spec.HorizonKind
        @test Spec.ZERO_SUM isa Spec.PayoffKind
        @test Spec.FULL_STATE_OBSERVATION isa Spec.ObservationKind
        @test Spec.SHARED_REWARD isa Spec.RewardSharing
    end

    @testset "GameSpec defaults" begin
        s = Spec.GameSpec()

        @test s.horizon_kind == Spec.EPISODIC
        @test s.payoff_kind == Spec.UNKNOWN_PAYOFF
        @test s.max_steps === nothing
        @test s.default_discount === nothing
        @test s.perfect_information === nothing
        @test s.stochastic === nothing
        @test s.simultaneous_moves === nothing
        @test s.observation_kind == Spec.UNKNOWN_OBSERVATION
        @test s.cooperative === nothing
        @test s.reward_sharing == Spec.UNKNOWN_REWARD_SHARING
    end

    @testset "game_spec default fallback" begin
        struct UnspecifiedGame <: Kernel.AbstractGame{1,Float64} end
        s = Spec.game_spec(UnspecifiedGame())

        @test s isa Spec.GameSpec
        @test s == Spec.GameSpec()
    end

    @testset "custom game_spec specializations are returned" begin
        s = Spec.game_spec(DecisionGame())
        @test s.perfect_information === true
        @test s.stochastic === false

        s2 = Spec.game_spec(SimultaneousGame())
        @test s2.perfect_information === false
        @test s2.stochastic === true
        @test s2.reward_sharing == Spec.SHARED_REWARD
    end
end