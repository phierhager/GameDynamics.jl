using Test
using Random

using GameLab.Spaces
using GameLab.LocalStrategies
using GameLab.BayesianPriors

@testset "BayesianPriors" begin
    @testset "CommonPrior canonicalization and probabilities" begin
        prior = BayesianPriors.CommonPrior(
            ((:H, :L), (:H, :L), (:L, :H)),
            (0.2, 0.3, 0.5),
        )

        @test length(prior.support_profiles) == 2
        @test prior.support_profiles[1] == (:H, :L)
        @test prior.support_profiles[2] == (:L, :H)
        @test isapprox(prior.probs[1], 0.5)
        @test isapprox(prior.probs[2], 0.5)

        @test isapprox(BayesianPriors.prior_probability(prior, (:H, :L)), 0.5)
        @test isapprox(BayesianPriors.prior_probability(prior, (:L, :H)), 0.5)
        @test isapprox(BayesianPriors.prior_probability(prior, (:L, :L)), 0.0)
    end

    @testset "CommonPrior input validation" begin
        @test_throws ArgumentError BayesianPriors.CommonPrior(((:H,),), (0.5, 0.5))
        @test_throws ArgumentError BayesianPriors.CommonPrior((), ())
    end

    @testset "CommonPrior marginals and type spaces" begin
        prior = BayesianPriors.CommonPrior(
            ((:A, :X), (:A, :Y), (:B, :Y)),
            (0.25, 0.25, 0.5),
        )

        @test isapprox(BayesianPriors.marginal_probability(prior, 1, :A), 0.5)
        @test isapprox(BayesianPriors.marginal_probability(prior, 1, :B), 0.5)
        @test isapprox(BayesianPriors.marginal_probability(prior, 2, :X), 0.25)
        @test isapprox(BayesianPriors.marginal_probability(prior, 2, :Y), 0.75)

        s1 = BayesianPriors.marginal_type_space(prior, 1)
        s2 = BayesianPriors.type_space(prior, 2)

        @test s1 isa Spaces.FiniteSpace
        @test s2 isa Spaces.FiniteSpace
        @test s1.elements == (:A, :B)
        @test s2.elements == (:X, :Y)

        @test_throws ArgumentError BayesianPriors.marginal_probability(prior, 3, :Z)
        @test_throws ArgumentError BayesianPriors.marginal_type_space(prior, 3)
    end

    @testset "CommonPrior sampling" begin
        prior = BayesianPriors.CommonPrior(
            ((:H, :L), (:L, :H)),
            (0.3, 0.7),
        )
        rng = MersenneTwister(1)
        x = BayesianPriors.sample_type_profile(prior, rng)
        @test x in prior.support_profiles
    end

    @testset "IndependentPrior constructor and accessors" begin
        spaces = (
            Spaces.FiniteSpace((:A, :B)),
            Spaces.FiniteSpace((:X, :Y)),
        )
        marginals = (
            LocalStrategies.FiniteMixedStrategy((:A, :B), (0.25, 0.75)),
            LocalStrategies.FiniteMixedStrategy((:X, :Y), (0.6, 0.4)),
        )

        prior = BayesianPriors.IndependentPrior(spaces, marginals)

        @test BayesianPriors.type_space(prior, 1) === spaces[1]
        @test BayesianPriors.marginal_type_space(prior, 2) === spaces[2]
        @test isapprox(BayesianPriors.marginal_probability(prior, 1, :A), 0.25)
        @test isapprox(BayesianPriors.marginal_probability(prior, 2, :Y), 0.4)

        @test_throws ArgumentError BayesianPriors.IndependentPrior((spaces[1],), marginals)
        @test_throws ArgumentError BayesianPriors.IndependentPrior((), ())
    end

    @testset "IndependentPrior full-profile probability and sampling" begin
        spaces = (
            Spaces.FiniteSpace((:A, :B)),
            Spaces.FiniteSpace((:X, :Y)),
        )
        marginals = (
            LocalStrategies.FiniteMixedStrategy((:A, :B), (0.25, 0.75)),
            LocalStrategies.FiniteMixedStrategy((:X, :Y), (0.6, 0.4)),
        )

        prior = BayesianPriors.IndependentPrior(spaces, marginals)
        @test isapprox(BayesianPriors.prior_probability(prior, (:A, :X)), 0.25 * 0.6)
        @test isapprox(BayesianPriors.prior_probability(prior, (:B, :Y)), 0.75 * 0.4)

        rng = MersenneTwister(2)
        prof = BayesianPriors.sample_type_profile(prior, rng)
        @test length(prof) == 2
        @test prof[1] in (:A, :B)
        @test prof[2] in (:X, :Y)

        @test_throws ArgumentError BayesianPriors.prior_probability(prior, (:A,))
        @test_throws ArgumentError BayesianPriors.marginal_probability(prior, 3, :Q)
    end
end