using Test
using Random
using GameLab.GameValidation
using GameLab.POSGValidation
using GameLab.DecPOMDPValidation
using GameLab.Kernel
using GameLab.Spec
using .TestMockGames

@testset "Validation" begin
    @testset "generic validate_game passes for sane mock game" begin
        g = DecisionGame()
        rep = GameValidation.validate_game(g)

        @test rep.family == :game
        @test rep.valid
        @test !isempty(rep.sections)
        @test GameValidation.is_valid_game(g)
        @test GameValidation.require_valid_game(g) === g
    end

    @testset "generic validate_game catches bad simultaneous ordering" begin
        g = BadSimultaneousGame()
        rep = GameValidation.validate_game(g)

        @test !rep.valid
        summary = GameValidation.summarize_failures(rep)
        @test occursin("simultaneous_active_players_ascending", summary)

        @test_throws ArgumentError GameValidation.require_valid_game(g)
    end

    @testset "generic validate_game catches invalid metadata" begin
        struct BadSpecGame <: Kernel.AbstractGame{2,Tuple{Float64,Float64}} end
        struct BadSpecState <: Kernel.AbstractState end

        Kernel.init_state(::BadSpecGame, rng::AbstractRNG = Random.default_rng()) = BadSpecState()
        Kernel.node_kind(::BadSpecGame, ::BadSpecState) = Kernel.DECISION
        Kernel.current_player(::BadSpecGame, ::BadSpecState) = 1

        Spec.game_spec(::BadSpecGame) = Spec.GameSpec(
            default_discount = 1.5,
            max_steps = 0,
        )

        rep = GameValidation.validate_game(BadSpecGame())
        @test !rep.valid

        summary = GameValidation.summarize_failures(rep)
        @test occursin("default_discount_valid", summary)
        @test occursin("max_steps_valid", summary)
    end

    @testset "POSG validation" begin
        g = SimultaneousGame()
        rep = POSGValidation.validate_posg(g)

        @test rep.family == :posg
        @test rep.valid
        @test POSGValidation.is_valid_posg(g)
        @test POSGValidation.require_valid_posg(g) === g
    end

    @testset "DecPOMDP validation" begin
        g = SimultaneousGame()
        rep = DecPOMDPValidation.validate_decpomdp(g)

        @test rep.family == :decpomdp
        @test rep.valid
        @test DecPOMDPValidation.is_valid_decpomdp(g)
        @test DecPOMDPValidation.require_valid_decpomdp(g) === g
    end

    @testset "ValidationReport show/summarize are stable enough" begin
        g = BadSimultaneousGame()
        rep = GameValidation.validate_game(g)

        shown = sprint(show, rep)
        @test occursin("ValidationReport(", shown)
        @test occursin("family=game", shown)

        summary = GameValidation.summarize_failures(rep)
        @test summary isa String
        @test !isempty(summary)
    end
end