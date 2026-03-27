using Test
using Random
using GameLab.Kernel
using .TestMockGames

@testset "Kernel" begin
    @testset "basic game metadata" begin
        g = DecisionGame()
        @test Kernel.num_players(g) == 2
        @test Tuple(Kernel.player_ids(g)) == (1, 2)
        @test Kernel.reward_type(typeof(g)) == Tuple{Float64,Float64}
    end

    @testset "JointAction constructors and indexing" begin
        ja1 = Kernel.joint_action((1, 2, 3))
        ja2 = Kernel.joint_action([1, 2, 3])
        ja3 = Kernel.joint_action(1, 2, 3)

        @test Tuple(ja1) == (1, 2, 3)
        @test Tuple(ja2) == (1, 2, 3)
        @test Tuple(ja3) == (1, 2, 3)

        @test length(ja1) == 3
        @test ja1[2] == 2
        @test firstindex(ja1) == 1
        @test lastindex(ja1) == 3
        @test collect(ja1) == [1, 2, 3]
    end

    @testset "acting_players and only_acting_player for decision nodes" begin
        g = DecisionGame()
        s = Kernel.init_state(g)

        @test Kernel.node_kind(g, s) == Kernel.DECISION
        @test Kernel.acting_players(g, s) == (2,)
        @test Kernel.only_acting_player(g, s) == 2
    end

    @testset "acting_players for simultaneous nodes" begin
        g = SimultaneousGame()
        s = Kernel.init_state(g)

        @test Kernel.node_kind(g, s) == Kernel.SIMULTANEOUS
        @test Tuple(Kernel.acting_players(g, s)) == (1, 3)
        @test_throws ArgumentError Kernel.only_acting_player(g, s)
    end

    @testset "action_for_player" begin
        g = SimultaneousGame()
        s = Kernel.init_state(g)
        ja = Kernel.joint_action(10, 21)

        @test Kernel.action_for_player(g, s, ja, 1) == 10
        @test Kernel.action_for_player(g, s, ja, 3) == 21
        @test Kernel.action_for_player(g, s, ja, 2) === nothing
    end

    @testset "validate_joint_action accepts valid aligned action tuples" begin
        g = SimultaneousGame()
        s = Kernel.init_state(g)
        ja = Kernel.joint_action(10, 20)

        @test Kernel.validate_joint_action(g, s, ja) === ja
    end

    @testset "validate_joint_action rejects wrong arity" begin
        g = SimultaneousGame()
        s = Kernel.init_state(g)

        err = ArgumentError
        @test_throws err Kernel.validate_joint_action(g, s, Kernel.joint_action(10))
        @test_throws err Kernel.validate_joint_action(g, s, Kernel.joint_action(10, 20, 30))
    end

    @testset "validate_joint_action rejects illegal actions" begin
        g = SimultaneousGame()
        s = Kernel.init_state(g)

        @test_throws ArgumentError Kernel.validate_joint_action(g, s, Kernel.joint_action(999, 20))
        @test_throws ArgumentError Kernel.validate_joint_action(g, s, Kernel.joint_action(10, 999))
    end

    @testset "validate_joint_action rejects non-simultaneous nodes" begin
        g = DecisionGame()
        s = Kernel.init_state(g)

        @test_throws ArgumentError Kernel.validate_joint_action(g, s, Kernel.joint_action(1))
    end

    @testset "validate_joint_action enforces strictly ascending active_players" begin
        g = BadSimultaneousGame()
        s = Kernel.init_state(g)

        @test_throws ArgumentError Kernel.validate_joint_action(g, s, Kernel.joint_action(7, 5))
    end

    @testset "is_terminal defaults from node_kind" begin
        struct TerminalGame <: Kernel.AbstractGame{1,Float64} end
        struct TerminalState <: Kernel.AbstractState end

        Kernel.init_state(::TerminalGame, rng::AbstractRNG = Random.default_rng()) = TerminalState()
        Kernel.node_kind(::TerminalGame, ::TerminalState) = Kernel.TERMINAL

        g = TerminalGame()
        s = Kernel.init_state(g)
        @test Kernel.is_terminal(g, s)
    end

    @testset "default action mode / mask support" begin
        @test Kernel.action_mode(DecisionGame) == Kernel.ExplicitActions
        @test Kernel.has_action_mask(DecisionGame) == false
    end
end