using Test

using GameLab
using GameLab.DecisionRuleInternalUtils

@testset "DecisionRuleInternalUtils" begin
    @testset "normalize_probs on tuple" begin
        # Arrange
        probs = (2, 6, 2)

        # Act
        normalized = DecisionRuleInternalUtils.normalize_probs(probs)

        # Assert
        @test normalized == (0.2, 0.6, 0.2)
        @test sum(normalized) ≈ 1.0
        @test normalized isa NTuple{3,Float64}
    end

    @testset "normalize_probs on vector" begin
        # Arrange
        probs = [1, 3]

        # Act
        normalized = DecisionRuleInternalUtils.normalize_probs(probs)

        # Assert
        @test normalized == [0.25, 0.75]
        @test sum(normalized) ≈ 1.0
        @test eltype(normalized) == Float64
    end

    @testset "normalize_probs rejects invalid inputs" begin
        # Arrange / Act / Assert
        @test_throws ArgumentError DecisionRuleInternalUtils.normalize_probs(())
        @test_throws ArgumentError DecisionRuleInternalUtils.normalize_probs(Int[])
        @test_throws ArgumentError DecisionRuleInternalUtils.normalize_probs((0, 0))
        @test_throws ArgumentError DecisionRuleInternalUtils.normalize_probs([0.0, 0.0])
        @test_throws ArgumentError DecisionRuleInternalUtils.normalize_probs((-1, 2))
        @test_throws ArgumentError DecisionRuleInternalUtils.normalize_probs([1.0, -0.5])
    end

    @testset "canonicalize_support_probs merges duplicates and preserves first-seen order" begin
        # Arrange
        actions = [:a, :b, :a, :c, :b]
        probs = (1, 2, 3, 4, 10)

        # Act
        acts, ps = DecisionRuleInternalUtils.canonicalize_support_probs(actions, probs)

        # Assert
        @test acts == (:a, :b, :c)
        @test ps[1] ≈ 4 / 20
        @test ps[2] ≈ 12 / 20
        @test ps[3] ≈ 4 / 20
        @test sum(ps) ≈ 1.0
    end

    @testset "canonicalize_support_probs rejects invalid lengths and empty support" begin
        # Arrange / Act / Assert
        @test_throws ArgumentError DecisionRuleInternalUtils.canonicalize_support_probs((:a,), (0.5, 0.5))
        @test_throws ArgumentError DecisionRuleInternalUtils.canonicalize_support_probs((), ())
    end

    @testset "canonicalize_joint_tuple_probs merges duplicate tuples" begin
        # Arrange
        joint_support = ((:L, :U), (:R, :D), (:L, :U))
        probs = (0.2, 0.3, 0.5)

        # Act
        tuples, ps = DecisionRuleInternalUtils.canonicalize_joint_tuple_probs(joint_support, probs)

        # Assert
        @test tuples == ((:L, :U), (:R, :D))
        @test ps[1] ≈ 0.7
        @test ps[2] ≈ 0.3
        @test sum(ps) ≈ 1.0
    end

    @testset "canonicalize_joint_tuple_probs rejects invalid inputs" begin
        # Arrange / Act / Assert
        @test_throws ArgumentError DecisionRuleInternalUtils.canonicalize_joint_tuple_probs((), ())
        @test_throws ArgumentError DecisionRuleInternalUtils.canonicalize_joint_tuple_probs(((1, 2),), (0.1, 0.9))
    end
end