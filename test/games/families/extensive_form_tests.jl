using Test
using Random

using GameLab.Kernel
using GameLab.ExtensiveForm
using GameLab.LocalStrategies

@testset "ExtensiveForm behavior helpers" begin
    struct EFState <: Kernel.AbstractState
        kind::Kernel.NodeKind
    end

    struct DecisionEFGame <: Kernel.AbstractGame{2,Tuple{Float64,Float64}} end
    struct SimultaneousEFGame <: Kernel.AbstractGame{2,Tuple{Float64,Float64}} end
    struct ChanceEFGame <: Kernel.AbstractGame{2,Tuple{Float64,Float64}} end
    struct TerminalEFGame <: Kernel.AbstractGame{2,Tuple{Float64,Float64}} end

    Kernel.init_state(::DecisionEFGame, rng::AbstractRNG=Random.default_rng()) = EFState(Kernel.DECISION)
    Kernel.init_state(::SimultaneousEFGame, rng::AbstractRNG=Random.default_rng()) = EFState(Kernel.SIMULTANEOUS)
    Kernel.init_state(::ChanceEFGame, rng::AbstractRNG=Random.default_rng()) = EFState(Kernel.CHANCE)
    Kernel.init_state(::TerminalEFGame, rng::AbstractRNG=Random.default_rng()) = EFState(Kernel.TERMINAL)

    Kernel.node_kind(::DecisionEFGame, s::EFState) = s.kind
    Kernel.node_kind(::SimultaneousEFGame, s::EFState) = s.kind
    Kernel.node_kind(::ChanceEFGame, s::EFState) = s.kind
    Kernel.node_kind(::TerminalEFGame, s::EFState) = s.kind

    Kernel.current_player(::DecisionEFGame, ::EFState) = 2
    Kernel.active_players(::SimultaneousEFGame, ::EFState) = (1, 2)

    Kernel.legal_actions(::DecisionEFGame, ::EFState, p::Int) = p == 2 ? (10, 11) : ()
    Kernel.legal_actions(::SimultaneousEFGame, ::EFState, p::Int) = p == 1 ? (1, 2) : (5, 6)

    Kernel.observe(::DecisionEFGame, s::EFState, p::Int) = (:obs, s.kind, p)
    Kernel.observe(::SimultaneousEFGame, s::EFState, p::Int) = (:obs, s.kind, p)

    struct ConstantActionStrategy{A} <: StrategyInterface.AbstractStrategy
        a::A
    end

    StrategyInterface.local_strategy(s::ConstantActionStrategy, infoset) = s
    StrategyInterface.sample_action(s::ConstantActionStrategy, infoset, rng::AbstractRNG=Random.default_rng()) = s.a
    StrategyInterface.action_probability(s::ConstantActionStrategy, infoset, action) = action == s.a ? 1.0 : 0.0

    @testset "local behavior helpers" begin
        g = DecisionEFGame()
        s = Kernel.init_state(g)
        strat = ConstantActionStrategy(11)

        @test ExtensiveForm.local_behavior(strat, g, s, 2) === strat
        @test ExtensiveForm.behavior_action_probability(strat, g, s, 2, 11) == 1.0
        @test ExtensiveForm.sample_behavior_action(strat, g, s, 2, MersenneTwister(1)) == 11
        @test ExtensiveForm.is_behavior_defined(strat, g, s, 2)
    end

    @testset "sample_behavior_profile_action at decision node" begin
        g = DecisionEFGame()
        s = Kernel.init_state(g)
        prof = (ConstantActionStrategy(1), ConstantActionStrategy(10))

        a = ExtensiveForm.sample_behavior_profile_action(prof, g, s, MersenneTwister(2))
        @test a == 10
    end

    @testset "sample_behavior_profile_action rejects illegal decision action" begin
        g = DecisionEFGame()
        s = Kernel.init_state(g)
        prof = (ConstantActionStrategy(1), ConstantActionStrategy(99))

        @test_throws ArgumentError ExtensiveForm.sample_behavior_profile_action(prof, g, s, MersenneTwister(3))
    end

    @testset "sample_behavior_profile_action at simultaneous node" begin
        g = SimultaneousEFGame()
        s = Kernel.init_state(g)
        prof = (ConstantActionStrategy(2), ConstantActionStrategy(6))

        ja = ExtensiveForm.sample_behavior_profile_action(prof, g, s, MersenneTwister(4))
        @test ja isa Kernel.JointAction
        @test Tuple(ja) == (2, 6)
    end

    @testset "sample_behavior_profile_action at chance and terminal nodes" begin
        cg = ChanceEFGame()
        cs = Kernel.init_state(cg)
        prof = (ConstantActionStrategy(1), ConstantActionStrategy(2))

        @test ExtensiveForm.sample_behavior_profile_action(prof, cg, cs, MersenneTwister(5)) isa Kernel.SampleChance

        tg = TerminalEFGame()
        ts = Kernel.init_state(tg)
        @test_throws ArgumentError ExtensiveForm.sample_behavior_profile_action(prof, tg, ts, MersenneTwister(6))
    end

    @testset "sample_behavior_profile_action rejects profile size mismatch" begin
        g = DecisionEFGame()
        s = Kernel.init_state(g)
        prof = (ConstantActionStrategy(10),)

        @test_throws ArgumentError ExtensiveForm.sample_behavior_profile_action(prof, g, s)
    end
end