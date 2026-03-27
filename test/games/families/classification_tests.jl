using Test
using GameLab.Classification
using GameLab.Kernel
using .TestMockGames

@testset "Classification" begin
    @testset "conservative defaults" begin
        g = DecisionGame()

        @test !Classification.is_normal_form(g)
        @test !Classification.is_extensive_form(g)
        @test !Classification.is_bayesian_game(g)
        @test !Classification.is_repeated_game(g)
        @test !Classification.is_complete_information(g)
        @test !Classification.is_incomplete_information(g)
        @test !Classification.is_signaling_game(g)
        @test !Classification.is_anonymous_game(g)
        @test !Classification.is_potential_game(g)
        @test !Classification.is_network_game(g)
        @test !Classification.is_stackelberg_game(g)
        @test !Classification.is_hierarchical_game(g)
    end

    @testset "metadata-derived predicates" begin
        g1 = DecisionGame()
        @test Classification.is_perfect_information(g1)
        @test !Classification.is_imperfect_information(g1)
        @test !Classification.is_stochastic_game(g1)
        @test Classification.is_zero_sum(g1)
        @test !Classification.is_general_sum(g1)
        @test Classification.is_noncooperative(g1)

        g2 = SimultaneousGame()
        @test !Classification.is_perfect_information(g2)
        @test Classification.is_imperfect_information(g2)
        @test Classification.is_stochastic_game(g2)
        @test !Classification.is_zero_sum(g2)
        @test !Classification.is_noncooperative(g2)
    end

    @testset "POSG and DecPOMDP heuristics" begin
        g = SimultaneousGame()

        @test Classification.is_posg(g)
        @test Classification.is_decpomdp(g)
    end

    @testset "DecPOMDP implies POSG in heuristics" begin
        g = SimultaneousGame()
        @test !Classification.is_decpomdp(g) || Classification.is_posg(g)
    end
end