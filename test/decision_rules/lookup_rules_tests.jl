using Test
using Random

using GameLab
using GameLab.DecisionRulesInterface
using GameLab.DirectDecisionRules
using GameLab.LookupDecisionRules

@testset "LookupDecisionRules" begin
    @testset "CallableLookupRule" begin
        @testset "constructor sets context kind and default internal state" begin
            # Arrange
            f = ctx -> DirectDecisionRules.DeterministicDecisionRule(ctx)

            # Act
            rule = LookupDecisionRules.CallableLookupRule(
                DecisionRulesInterface.CustomContext,
                f,
            )

            # Assert
            @test DecisionRulesInterface.context_kind(rule) isa DecisionRulesInterface.CustomContext
            @test DecisionRulesInterface.internal_state_class(rule) isa DecisionRulesInterface.Stateless
        end

        @testset "constructor preserves explicit internal state" begin
            # Arrange
            f = ctx -> DirectDecisionRules.DeterministicDecisionRule(ctx)

            # Act
            rule = LookupDecisionRules.CallableLookupRule(
                DecisionRulesInterface.ObservationContext,
                f;
                internal_state = DecisionRulesInterface.Stateful(),
            )

            # Assert
            @test DecisionRulesInterface.context_kind(rule) isa DecisionRulesInterface.ObservationContext
            @test DecisionRulesInterface.internal_state_class(rule) isa DecisionRulesInterface.Stateful
        end

        @testset "local_rule returns callable-produced local decision rule" begin
            # Arrange
            rule = LookupDecisionRules.CallableLookupRule(
                DecisionRulesInterface.ObservationContext,
                obs -> DirectDecisionRules.DeterministicDecisionRule(obs + 1),
            )

            # Act
            local_r = DecisionRulesInterface.local_rule(rule, 4)

            # Assert
            @test local_r isa DirectDecisionRules.DeterministicDecisionRule
            @test local_r.action == 5
        end

        @testset "sample_action delegates through local rule" begin
            # Arrange
            rule = LookupDecisionRules.CallableLookupRule(
                DecisionRulesInterface.ObservationContext,
                obs -> DirectDecisionRules.DeterministicDecisionRule(obs + 10),
            )
            rng = MersenneTwister(11)

            # Act
            sampled = DecisionRulesInterface.sample_action(rule, 2, rng)

            # Assert
            @test sampled == 12
        end

        @testset "action_probability delegates through local rule" begin
            # Arrange
            rule = LookupDecisionRules.CallableLookupRule(
                DecisionRulesInterface.CustomContext,
                ctx -> DirectDecisionRules.FiniteMixedDecisionRule((ctx, :other), (0.8, 0.2)),
            )

            # Act
            p_hit = DecisionRulesInterface.action_probability(rule, :go, :go)
            p_miss = DecisionRulesInterface.action_probability(rule, :go, :other)

            # Assert
            @test p_hit ≈ 0.8
            @test p_miss ≈ 0.2
        end

        @testset "action_density delegates through local rule" begin
            # Arrange
            density_rule = DirectDecisionRules.SamplerDensityDecisionRule(
                r -> 0.25,
                x -> x^2,
                0.0:0.1:1.0,
            )
            rule = LookupDecisionRules.CallableLookupRule(
                DecisionRulesInterface.CustomContext,
                _ -> density_rule,
            )

            # Act
            density = DecisionRulesInterface.action_density(rule, :ctx, 3.0)

            # Assert
            @test density == 9.0
        end

        @testset "local_rule throws when callable does not return a decision rule" begin
            # Arrange
            rule = LookupDecisionRules.CallableLookupRule(
                DecisionRulesInterface.ObservationContext,
                _ -> 123,
            )

            # Act / Assert
            @test_throws ArgumentError DecisionRulesInterface.local_rule(rule, :ctx)
        end
    end

    @testset "TableLookupRule" begin
        @testset "constructor sets context kind and default internal state" begin
            # Arrange
            table = Dict(
                :left => DirectDecisionRules.DeterministicDecisionRule(:L),
                :right => DirectDecisionRules.DeterministicDecisionRule(:R),
            )

            # Act
            rule = LookupDecisionRules.TableLookupRule(
                DecisionRulesInterface.CustomContext,
                table,
            )

            # Assert
            @test DecisionRulesInterface.context_kind(rule) isa DecisionRulesInterface.CustomContext
            @test DecisionRulesInterface.internal_state_class(rule) isa DecisionRulesInterface.Stateless
        end

        @testset "constructor preserves explicit internal state" begin
            # Arrange
            table = Dict(
                1 => DirectDecisionRules.DeterministicDecisionRule(:a),
                2 => DirectDecisionRules.DeterministicDecisionRule(:b),
            )

            # Act
            rule = LookupDecisionRules.TableLookupRule(
                DecisionRulesInterface.StateContext,
                table;
                internal_state = DecisionRulesInterface.FiniteStateController(),
            )

            # Assert
            @test DecisionRulesInterface.context_kind(rule) isa DecisionRulesInterface.StateContext
            @test DecisionRulesInterface.internal_state_class(rule) isa DecisionRulesInterface.FiniteStateController
        end

        @testset "local_rule returns stored rule for present key" begin
            # Arrange
            left_rule = DirectDecisionRules.DeterministicDecisionRule(:L)
            right_rule = DirectDecisionRules.DeterministicDecisionRule(:R)
            table = Dict(:left => left_rule, :right => right_rule)
            rule = LookupDecisionRules.TableLookupRule(
                DecisionRulesInterface.CustomContext,
                table,
            )

            # Act
            local_r = DecisionRulesInterface.local_rule(rule, :right)

            # Assert
            @test local_r === right_rule
        end

        @testset "sample_action delegates through stored local rule" begin
            # Arrange
            table = Dict(
                :a => DirectDecisionRules.DeterministicDecisionRule(10),
                :b => DirectDecisionRules.DeterministicDecisionRule(20),
            )
            rule = LookupDecisionRules.TableLookupRule(
                DecisionRulesInterface.CustomContext,
                table,
            )
            rng = MersenneTwister(7)

            # Act
            sampled = DecisionRulesInterface.sample_action(rule, :b, rng)

            # Assert
            @test sampled == 20
        end

        @testset "action_probability delegates through stored local rule" begin
            # Arrange
            table = Dict(
                :x => DirectDecisionRules.FiniteMixedDecisionRule((:a, :b), (0.25, 0.75)),
            )
            rule = LookupDecisionRules.TableLookupRule(
                DecisionRulesInterface.CustomContext,
                table,
            )

            # Act
            p_a = DecisionRulesInterface.action_probability(rule, :x, :a)
            p_b = DecisionRulesInterface.action_probability(rule, :x, :b)
            p_c = DecisionRulesInterface.action_probability(rule, :x, :c)

            # Assert
            @test p_a ≈ 0.25
            @test p_b ≈ 0.75
            @test p_c == 0.0
        end

        @testset "action_density delegates through stored local rule" begin
            # Arrange
            density_rule = DirectDecisionRules.SamplerDensityDecisionRule(
                r -> 0.5,
                x -> x + 2,
                0.0:0.5:2.0,
            )
            table = Dict(:k => density_rule)
            rule = LookupDecisionRules.TableLookupRule(
                DecisionRulesInterface.CustomContext,
                table,
            )

            # Act
            density = DecisionRulesInterface.action_density(rule, :k, 3.0)

            # Assert
            @test density == 5.0
        end

        @testset "local_rule throws KeyError for missing context" begin
            # Arrange
            table = Dict(:a => DirectDecisionRules.DeterministicDecisionRule(1))
            rule = LookupDecisionRules.TableLookupRule(
                DecisionRulesInterface.CustomContext,
                table,
            )

            # Act / Assert
            @test_throws KeyError DecisionRulesInterface.local_rule(rule, :missing)
        end
    end

    @testset "DenseLookupRule" begin
        @testset "constructor sets context kind and default internal state" begin
            # Arrange
            table = (
                DirectDecisionRules.DeterministicDecisionRule(:a),
                DirectDecisionRules.DeterministicDecisionRule(:b),
            )

            # Act
            rule = LookupDecisionRules.DenseLookupRule(
                DecisionRulesInterface.StateContext,
                table,
            )

            # Assert
            @test DecisionRulesInterface.context_kind(rule) isa DecisionRulesInterface.StateContext
            @test DecisionRulesInterface.internal_state_class(rule) isa DecisionRulesInterface.Stateless
        end

        @testset "constructor preserves explicit internal state" begin
            # Arrange
            table = (
                DirectDecisionRules.DeterministicDecisionRule(1),
                DirectDecisionRules.DeterministicDecisionRule(2),
            )

            # Act
            rule = LookupDecisionRules.DenseLookupRule(
                DecisionRulesInterface.StateContext,
                table;
                internal_state = DecisionRulesInterface.Stateful(),
            )

            # Assert
            @test DecisionRulesInterface.context_kind(rule) isa DecisionRulesInterface.StateContext
            @test DecisionRulesInterface.internal_state_class(rule) isa DecisionRulesInterface.Stateful
        end

        @testset "local_rule returns tuple-indexed local rule" begin
            # Arrange
            first_rule = DirectDecisionRules.DeterministicDecisionRule(:left)
            second_rule = DirectDecisionRules.DeterministicDecisionRule(:right)
            rule = LookupDecisionRules.DenseLookupRule(
                DecisionRulesInterface.StateContext,
                (first_rule, second_rule),
            )

            # Act
            local1 = DecisionRulesInterface.local_rule(rule, 1)
            local2 = DecisionRulesInterface.local_rule(rule, 2)

            # Assert
            @test local1 === first_rule
            @test local2 === second_rule
        end

        @testset "sample_action delegates through indexed local rule" begin
            # Arrange
            rule = LookupDecisionRules.DenseLookupRule(
                DecisionRulesInterface.StateContext,
                (
                    DirectDecisionRules.DeterministicDecisionRule(:x),
                    DirectDecisionRules.DeterministicDecisionRule(:y),
                ),
            )
            rng = MersenneTwister(13)

            # Act
            sampled = DecisionRulesInterface.sample_action(rule, 2, rng)

            # Assert
            @test sampled == :y
        end

        @testset "action_probability delegates through indexed local rule" begin
            # Arrange
            rule = LookupDecisionRules.DenseLookupRule(
                DecisionRulesInterface.StateContext,
                (
                    DirectDecisionRules.FiniteMixedDecisionRule((:a, :b), (0.6, 0.4)),
                    DirectDecisionRules.FiniteMixedDecisionRule((:c, :d), (0.1, 0.9)),
                ),
            )

            # Act
            p1 = DecisionRulesInterface.action_probability(rule, 1, :a)
            p2 = DecisionRulesInterface.action_probability(rule, 2, :d)

            # Assert
            @test p1 ≈ 0.6
            @test p2 ≈ 0.9
        end

        @testset "action_density delegates through indexed local rule" begin
            # Arrange
            density_rule_1 = DirectDecisionRules.SamplerDensityDecisionRule(
                r -> 0.1,
                x -> x,
                0:1,
            )
            density_rule_2 = DirectDecisionRules.SamplerDensityDecisionRule(
                r -> 0.2,
                x -> 3x,
                0:1,
            )
            rule = LookupDecisionRules.DenseLookupRule(
                DecisionRulesInterface.StateContext,
                (density_rule_1, density_rule_2),
            )

            # Act
            density = DecisionRulesInterface.action_density(rule, 2, 4.0)

            # Assert
            @test density == 12.0
        end

        @testset "local_rule throws BoundsError for invalid integer context" begin
            # Arrange
            rule = LookupDecisionRules.DenseLookupRule(
                DecisionRulesInterface.StateContext,
                (
                    DirectDecisionRules.DeterministicDecisionRule(:a),
                    DirectDecisionRules.DeterministicDecisionRule(:b),
                ),
            )

            # Act / Assert
            @test_throws BoundsError DecisionRulesInterface.local_rule(rule, 0)
            @test_throws BoundsError DecisionRulesInterface.local_rule(rule, 3)
        end
    end

    @testset "DenseVectorLookupRule" begin
        @testset "constructor sets context kind and default internal state" begin
            # Arrange
            table = [
                DirectDecisionRules.DeterministicDecisionRule(:a),
                DirectDecisionRules.DeterministicDecisionRule(:b),
            ]

            # Act
            rule = LookupDecisionRules.DenseVectorLookupRule(
                DecisionRulesInterface.StateContext,
                table,
            )

            # Assert
            @test DecisionRulesInterface.context_kind(rule) isa DecisionRulesInterface.StateContext
            @test DecisionRulesInterface.internal_state_class(rule) isa DecisionRulesInterface.Stateless
        end

        @testset "constructor preserves explicit internal state" begin
            # Arrange
            table = [
                DirectDecisionRules.DeterministicDecisionRule(1),
                DirectDecisionRules.DeterministicDecisionRule(2),
            ]

            # Act
            rule = LookupDecisionRules.DenseVectorLookupRule(
                DecisionRulesInterface.StateContext,
                table;
                internal_state = DecisionRulesInterface.FiniteStateController(),
            )

            # Assert
            @test DecisionRulesInterface.context_kind(rule) isa DecisionRulesInterface.StateContext
            @test DecisionRulesInterface.internal_state_class(rule) isa DecisionRulesInterface.FiniteStateController
        end

        @testset "local_rule returns vector-indexed local rule" begin
            # Arrange
            first_rule = DirectDecisionRules.DeterministicDecisionRule(:left)
            second_rule = DirectDecisionRules.DeterministicDecisionRule(:right)
            rule = LookupDecisionRules.DenseVectorLookupRule(
                DecisionRulesInterface.StateContext,
                [first_rule, second_rule],
            )

            # Act
            local1 = DecisionRulesInterface.local_rule(rule, 1)
            local2 = DecisionRulesInterface.local_rule(rule, 2)

            # Assert
            @test local1 === first_rule
            @test local2 === second_rule
        end

        @testset "sample_action delegates through indexed local rule" begin
            # Arrange
            rule = LookupDecisionRules.DenseVectorLookupRule(
                DecisionRulesInterface.StateContext,
                [
                    DirectDecisionRules.DeterministicDecisionRule(100),
                    DirectDecisionRules.DeterministicDecisionRule(200),
                ],
            )
            rng = MersenneTwister(21)

            # Act
            sampled = DecisionRulesInterface.sample_action(rule, 1, rng)

            # Assert
            @test sampled == 100
        end

        @testset "action_probability delegates through indexed local rule" begin
            # Arrange
            rule = LookupDecisionRules.DenseVectorLookupRule(
                DecisionRulesInterface.StateContext,
                [
                    DirectDecisionRules.FiniteMixedDecisionRule((:x, :y), (0.2, 0.8)),
                    DirectDecisionRules.FiniteMixedDecisionRule((:u, :v), (0.55, 0.45)),
                ],
            )

            # Act
            p1 = DecisionRulesInterface.action_probability(rule, 1, :y)
            p2 = DecisionRulesInterface.action_probability(rule, 2, :u)

            # Assert
            @test p1 ≈ 0.8
            @test p2 ≈ 0.55
        end

        @testset "action_density delegates through indexed local rule" begin
            # Arrange
            density_rule_1 = DirectDecisionRules.SamplerDensityDecisionRule(
                r -> 0.3,
                x -> x - 1,
                0:1,
            )
            density_rule_2 = DirectDecisionRules.SamplerDensityDecisionRule(
                r -> 0.4,
                x -> x / 2,
                0:1,
            )
            rule = LookupDecisionRules.DenseVectorLookupRule(
                DecisionRulesInterface.StateContext,
                [density_rule_1, density_rule_2],
            )

            # Act
            density = DecisionRulesInterface.action_density(rule, 2, 8.0)

            # Assert
            @test density == 4.0
        end

        @testset "local_rule throws BoundsError for invalid integer context" begin
            # Arrange
            rule = LookupDecisionRules.DenseVectorLookupRule(
                DecisionRulesInterface.StateContext,
                [DirectDecisionRules.DeterministicDecisionRule(:x)],
            )

            # Act / Assert
            @test_throws BoundsError DecisionRulesInterface.local_rule(rule, 0)
            @test_throws BoundsError DecisionRulesInterface.local_rule(rule, 2)
        end
    end
end