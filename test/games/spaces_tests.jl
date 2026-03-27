using Test
using Random
using GameLab.Spaces

@testset "Spaces" begin
    @testset "FiniteSpace" begin
        s1 = Spaces.FiniteSpace([:a, :b, :c])
        s2 = Spaces.FiniteSpace((1, 2, 3))

        @test Spaces.dimension(s1) === nothing
        @test Spaces.contains(s1, :a)
        @test !Spaces.contains(s1, :z)

        @test Spaces.contains(s2, 2)
        @test !Spaces.contains(s2, 5)

        @test_throws ArgumentError Spaces.FiniteSpace(Int[])
        @test_throws ArgumentError Spaces.FiniteSpace(())
    end

    @testset "IndexedDiscreteSpace" begin
        s = Spaces.IndexedDiscreteSpace(5)

        @test Spaces.dimension(s) == 5
        @test Spaces.contains(s, 1)
        @test Spaces.contains(s, 5)
        @test !Spaces.contains(s, 0)
        @test !Spaces.contains(s, 6)
        @test !Spaces.contains(s, 2.5)

        @test_throws ArgumentError Spaces.IndexedDiscreteSpace(0)
    end

    @testset "BoxSpace" begin
        s = Spaces.BoxSpace([0.0, -1.0], [1.0, 2.0])

        @test Spaces.dimension(s) == 2
        @test Spaces.contains(s, [0.5, 0.0])
        @test Spaces.contains(s, (1.0, -1.0))
        @test !Spaces.contains(s, [2.0, 0.0])
        @test !Spaces.contains(s, [0.5])
        @test !Spaces.contains(s, :bad)

        @test_throws ArgumentError Spaces.BoxSpace([0.0], [1.0, 2.0])
        @test_throws ArgumentError Spaces.BoxSpace([1.0, 0.0], [0.0, 2.0])
    end

    @testset "SimplexSpace" begin
        s = Spaces.SimplexSpace(3)

        @test Spaces.dimension(s) == 3
        @test Spaces.contains(s, [0.2, 0.3, 0.5])
        @test Spaces.contains(s, (1/3, 1/3, 1/3))
        @test !Spaces.contains(s, [0.2, 0.3, 0.6])
        @test !Spaces.contains(s, [0.2, -0.1, 0.9])
        @test !Spaces.contains(s, [0.5, 0.5])
        @test !Spaces.contains(s, "bad")

        @test_throws ArgumentError Spaces.SimplexSpace(0)
    end

    @testset "ProductSpace" begin
        s = Spaces.ProductSpace((
            Spaces.IndexedDiscreteSpace(3),
            Spaces.BoxSpace([0.0], [1.0]),
            Spaces.SimplexSpace(2),
        ))

        @test Spaces.dimension(s) == 3 + 1 + 2
        @test Spaces.contains(s, (2, [0.5], [0.25, 0.75]))
        @test !Spaces.contains(s, (4, [0.5], [0.25, 0.75]))
        @test !Spaces.contains(s, (2, [1.5], [0.25, 0.75]))
        @test !Spaces.contains(s, (2, [0.5], [0.25, 0.80]))
        @test !Spaces.contains(s, (2, [0.5]))
        @test !Spaces.contains(s, :bad)

        @test_throws ArgumentError Spaces.ProductSpace(())
    end

    @testset "ProductSpace dimension returns nothing if any factor is unknown" begin
        s = Spaces.ProductSpace((
            Spaces.FiniteSpace((:a, :b)),
            Spaces.IndexedDiscreteSpace(2),
        ))

        @test Spaces.dimension(s) === nothing
    end

    @testset "sampling" begin
        rng = MersenneTwister(1)

        fs = Spaces.FiniteSpace((10, 20, 30))
        @test Spaces.sample(rng, fs) in (10, 20, 30)

        ids = Spaces.IndexedDiscreteSpace(4)
        x = Spaces.sample(rng, ids)
        @test 1 <= x <= 4

        bs = Spaces.BoxSpace([0.0, 0.0], [1.0, 2.0])
        xb = Spaces.sample(rng, bs)
        @test Spaces.contains(bs, xb)

        ss = Spaces.SimplexSpace(4)
        xs = Spaces.sample(rng, ss)
        @test Spaces.contains(ss, xs)

        ps = Spaces.ProductSpace((ids, ss))
        xp = Spaces.sample(rng, ps)
        @test xp isa Tuple
        @test Spaces.contains(ps, xp)
    end
end