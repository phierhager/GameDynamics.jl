using Test
using GameLab.Enumerative
using GameLab.Kernel

@testset "Enumerative" begin
    struct BareGame <: Kernel.AbstractGame{1,Float64} end

    g = BareGame()

    @test_throws MethodError Enumerative.chance_outcomes(g, :s)
    @test_throws MethodError Enumerative.transition_kernel(g, :s, :a)
    @test_throws MethodError Enumerative.terminal_payoffs(g, :s)
end