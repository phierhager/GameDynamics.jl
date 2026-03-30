using Test
using Random

using GameLab
using GameLab.StrategyInterface
using GameLab.LocalStrategies
using GameLab.IndexedStrategies

@testset "IndexedStrategies" begin
    @testset "CallableIndexedStrategy" begin
        @testset "constructor sets context kind and default internal state" begin
            # Arrange
            f = ctx -> LocalStrategies.DeterministicStrategy(ctx)

            # Act
            strategy = IndexedStrategies.CallableIndexedStrategy(
                StrategyInterface.CustomContext,
                f,
            )

            # Assert
            @test StrategyInterface.context_kind(strategy) isa StrategyInterface.CustomContext
            @test StrategyInterface.internal_state_class(strategy) isa StrategyInterface.Stateless
        end

        @testset "constructor preserves explicit internal state" begin
            # Arrange
            f = ctx -> LocalStrategies.DeterministicStrategy(ctx)

            # Act
            strategy = IndexedStrategies.CallableIndexedStrategy(
                StrategyInterface.ObservationContext,
                f;
                internal_state = StrategyInterface.Stateful(),
            )

            # Assert
            @test StrategyInterface.context_kind(strategy) isa StrategyInterface.ObservationContext
            @test StrategyInterface.internal_state_class(strategy) isa StrategyInterface.Stateful
        end

        @testset "local_strategy returns callable-produced local strategy" begin
            # Arrange
            strategy = IndexedStrategies.CallableIndexedStrategy(
                StrategyInterface.ObservationContext,
                obs -> LocalStrategies.DeterministicStrategy(obs + 1),
            )

            # Act
            local_r = StrategyInterface.local_strategy(strategy, 4)

            # Assert
            @test local_r isa LocalStrategies.DeterministicStrategy
            @test local_r.action == 5
        end

        @testset "sample_action delegates through local strategy" begin
            # Arrange
            strategy = IndexedStrategies.CallableIndexedStrategy(
                StrategyInterface.ObservationContext,
                obs -> LocalStrategies.DeterministicStrategy(obs + 10),
            )
            rng = MersenneTwister(11)

            # Act
            sampled = StrategyInterface.sample_action(strategy, 2, rng)

            # Assert
            @test sampled == 12
        end

        @testset "action_probability delegates through local strategy" begin
            # Arrange
            strategy = IndexedStrategies.CallableIndexedStrategy(
                StrategyInterface.CustomContext,
                ctx -> LocalStrategies.FiniteMixedStrategy((ctx, :other), (0.8, 0.2)),
            )

            # Act
            p_hit = StrategyInterface.action_probability(strategy, :go, :go)
            p_miss = StrategyInterface.action_probability(strategy, :go, :other)

            # Assert
            @test p_hit ≈ 0.8
            @test p_miss ≈ 0.2
        end

        @testset "action_density delegates through local strategy" begin
            # Arrange
            density_rule = LocalStrategies.SamplerDensityStrategy(
                r -> 0.25,
                x -> x^2,
                0.0:0.1:1.0,
            )
            strategy = IndexedStrategies.CallableIndexedStrategy(
                StrategyInterface.CustomContext,
                _ -> density_rule,
            )

            # Act
            density = StrategyInterface.action_density(strategy, :ctx, 3.0)

            # Assert
            @test density == 9.0
        end

        @testset "local_strategy throws when callable does not return a strategy" begin
            # Arrange
            strategy = IndexedStrategies.CallableIndexedStrategy(
                StrategyInterface.ObservationContext,
                _ -> 123,
            )

            # Act / Assert
            @test_throws ArgumentError StrategyInterface.local_strategy(strategy, :ctx)
        end
    end

    @testset "TableIndexedStrategy" begin
        @testset "constructor sets context kind and default internal state" begin
            # Arrange
            table = Dict(
                :left => LocalStrategies.DeterministicStrategy(:L),
                :right => LocalStrategies.DeterministicStrategy(:R),
            )

            # Act
            strategy = IndexedStrategies.TableIndexedStrategy(
                StrategyInterface.CustomContext,
                table,
            )

            # Assert
            @test StrategyInterface.context_kind(strategy) isa StrategyInterface.CustomContext
            @test StrategyInterface.internal_state_class(strategy) isa StrategyInterface.Stateless
        end

        @testset "constructor preserves explicit internal state" begin
            # Arrange
            table = Dict(
                1 => LocalStrategies.DeterministicStrategy(:a),
                2 => LocalStrategies.DeterministicStrategy(:b),
            )

            # Act
            strategy = IndexedStrategies.TableIndexedStrategy(
                StrategyInterface.StateContext,
                table;
                internal_state = StrategyInterface.FiniteStateController(),
            )

            # Assert
            @test StrategyInterface.context_kind(strategy) isa StrategyInterface.StateContext
            @test StrategyInterface.internal_state_class(strategy) isa StrategyInterface.FiniteStateController
        end

        @testset "local_strategy returns stored strategy for present key" begin
            # Arrange
            left_rule = LocalStrategies.DeterministicStrategy(:L)
            right_rule = LocalStrategies.DeterministicStrategy(:R)
            table = Dict(:left => left_rule, :right => right_rule)
            strategy = IndexedStrategies.TableIndexedStrategy(
                StrategyInterface.CustomContext,
                table,
            )

            # Act
            local_r = StrategyInterface.local_strategy(strategy, :right)

            # Assert
            @test local_r === right_rule
        end

        @testset "sample_action delegates through stored local strategy" begin
            # Arrange
            table = Dict(
                :a => LocalStrategies.DeterministicStrategy(10),
                :b => LocalStrategies.DeterministicStrategy(20),
            )
            strategy = IndexedStrategies.TableIndexedStrategy(
                StrategyInterface.CustomContext,
                table,
            )
            rng = MersenneTwister(7)

            # Act
            sampled = StrategyInterface.sample_action(strategy, :b, rng)

            # Assert
            @test sampled == 20
        end

        @testset "action_probability delegates through stored local strategy" begin
            # Arrange
            table = Dict(
                :x => LocalStrategies.FiniteMixedStrategy((:a, :b), (0.25, 0.75)),
            )
            strategy = IndexedStrategies.TableIndexedStrategy(
                StrategyInterface.CustomContext,
                table,
            )

            # Act
            p_a = StrategyInterface.action_probability(strategy, :x, :a)
            p_b = StrategyInterface.action_probability(strategy, :x, :b)
            p_c = StrategyInterface.action_probability(strategy, :x, :c)

            # Assert
            @test p_a ≈ 0.25
            @test p_b ≈ 0.75
            @test p_c == 0.0
        end

        @testset "action_density delegates through stored local strategy" begin
            # Arrange
            density_rule = LocalStrategies.SamplerDensityStrategy(
                r -> 0.5,
                x -> x + 2,
                0.0:0.5:2.0,
            )
            table = Dict(:k => density_rule)
            strategy = IndexedStrategies.TableIndexedStrategy(
                StrategyInterface.CustomContext,
                table,
            )

            # Act
            density = StrategyInterface.action_density(strategy, :k, 3.0)

            # Assert
            @test density == 5.0
        end

        @testset "local_strategy throws KeyError for missing context" begin
            # Arrange
            table = Dict(:a => LocalStrategies.DeterministicStrategy(1))
            strategy = IndexedStrategies.TableIndexedStrategy(
                StrategyInterface.CustomContext,
                table,
            )

            # Act / Assert
            @test_throws KeyError StrategyInterface.local_strategy(strategy, :missing)
        end
    end

    @testset "DenseIndexedStrategy" begin
        @testset "constructor sets context kind and default internal state" begin
            # Arrange
            table = (
                LocalStrategies.DeterministicStrategy(:a),
                LocalStrategies.DeterministicStrategy(:b),
            )

            # Act
            strategy = IndexedStrategies.DenseIndexedStrategy(
                StrategyInterface.StateContext,
                table,
            )

            # Assert
            @test StrategyInterface.context_kind(strategy) isa StrategyInterface.StateContext
            @test StrategyInterface.internal_state_class(strategy) isa StrategyInterface.Stateless
        end

        @testset "constructor preserves explicit internal state" begin
            # Arrange
            table = (
                LocalStrategies.DeterministicStrategy(1),
                LocalStrategies.DeterministicStrategy(2),
            )

            # Act
            strategy = IndexedStrategies.DenseIndexedStrategy(
                StrategyInterface.StateContext,
                table;
                internal_state = StrategyInterface.Stateful(),
            )

            # Assert
            @test StrategyInterface.context_kind(strategy) isa StrategyInterface.StateContext
            @test StrategyInterface.internal_state_class(strategy) isa StrategyInterface.Stateful
        end

        @testset "local_strategy returns tuple-indexed local strategy" begin
            # Arrange
            first_rule = LocalStrategies.DeterministicStrategy(:left)
            second_rule = LocalStrategies.DeterministicStrategy(:right)
            strategy = IndexedStrategies.DenseIndexedStrategy(
                StrategyInterface.StateContext,
                (first_rule, second_rule),
            )

            # Act
            local1 = StrategyInterface.local_strategy(strategy, 1)
            local2 = StrategyInterface.local_strategy(strategy, 2)

            # Assert
            @test local1 === first_rule
            @test local2 === second_rule
        end

        @testset "sample_action delegates through indexed local strategy" begin
            # Arrange
            strategy = IndexedStrategies.DenseIndexedStrategy(
                StrategyInterface.StateContext,
                (
                    LocalStrategies.DeterministicStrategy(:x),
                    LocalStrategies.DeterministicStrategy(:y),
                ),
            )
            rng = MersenneTwister(13)

            # Act
            sampled = StrategyInterface.sample_action(strategy, 2, rng)

            # Assert
            @test sampled == :y
        end

        @testset "action_probability delegates through indexed local strategy" begin
            # Arrange
            strategy = IndexedStrategies.DenseIndexedStrategy(
                StrategyInterface.StateContext,
                (
                    LocalStrategies.FiniteMixedStrategy((:a, :b), (0.6, 0.4)),
                    LocalStrategies.FiniteMixedStrategy((:c, :d), (0.1, 0.9)),
                ),
            )

            # Act
            p1 = StrategyInterface.action_probability(strategy, 1, :a)
            p2 = StrategyInterface.action_probability(strategy, 2, :d)

            # Assert
            @test p1 ≈ 0.6
            @test p2 ≈ 0.9
        end

        @testset "action_density delegates through indexed local strategy" begin
            # Arrange
            density_rule_1 = LocalStrategies.SamplerDensityStrategy(
                r -> 0.1,
                x -> x,
                0:1,
            )
            density_rule_2 = LocalStrategies.SamplerDensityStrategy(
                r -> 0.2,
                x -> 3x,
                0:1,
            )
            strategy = IndexedStrategies.DenseIndexedStrategy(
                StrategyInterface.StateContext,
                (density_rule_1, density_rule_2),
            )

            # Act
            density = StrategyInterface.action_density(strategy, 2, 4.0)

            # Assert
            @test density == 12.0
        end

        @testset "local_strategy throws BoundsError for invalid integer context" begin
            # Arrange
            strategy = IndexedStrategies.DenseIndexedStrategy(
                StrategyInterface.StateContext,
                (
                    LocalStrategies.DeterministicStrategy(:a),
                    LocalStrategies.DeterministicStrategy(:b),
                ),
            )

            # Act / Assert
            @test_throws BoundsError StrategyInterface.local_strategy(strategy, 0)
            @test_throws BoundsError StrategyInterface.local_strategy(strategy, 3)
        end
    end

    @testset "DenseVectorIndexedStrategy" begin
        @testset "constructor sets context kind and default internal state" begin
            # Arrange
            table = [
                LocalStrategies.DeterministicStrategy(:a),
                LocalStrategies.DeterministicStrategy(:b),
            ]

            # Act
            strategy = IndexedStrategies.DenseVectorIndexedStrategy(
                StrategyInterface.StateContext,
                table,
            )

            # Assert
            @test StrategyInterface.context_kind(strategy) isa StrategyInterface.StateContext
            @test StrategyInterface.internal_state_class(strategy) isa StrategyInterface.Stateless
        end

        @testset "constructor preserves explicit internal state" begin
            # Arrange
            table = [
                LocalStrategies.DeterministicStrategy(1),
                LocalStrategies.DeterministicStrategy(2),
            ]

            # Act
            strategy = IndexedStrategies.DenseVectorIndexedStrategy(
                StrategyInterface.StateContext,
                table;
                internal_state = StrategyInterface.FiniteStateController(),
            )

            # Assert
            @test StrategyInterface.context_kind(strategy) isa StrategyInterface.StateContext
            @test StrategyInterface.internal_state_class(strategy) isa StrategyInterface.FiniteStateController
        end

        @testset "local_strategy returns vector-indexed local strategy" begin
            # Arrange
            first_rule = LocalStrategies.DeterministicStrategy(:left)
            second_rule = LocalStrategies.DeterministicStrategy(:right)
            strategy = IndexedStrategies.DenseVectorIndexedStrategy(
                StrategyInterface.StateContext,
                [first_rule, second_rule],
            )

            # Act
            local1 = StrategyInterface.local_strategy(strategy, 1)
            local2 = StrategyInterface.local_strategy(strategy, 2)

            # Assert
            @test local1 === first_rule
            @test local2 === second_rule
        end

        @testset "sample_action delegates through indexed local strategy" begin
            # Arrange
            strategy = IndexedStrategies.DenseVectorIndexedStrategy(
                StrategyInterface.StateContext,
                [
                    LocalStrategies.DeterministicStrategy(100),
                    LocalStrategies.DeterministicStrategy(200),
                ],
            )
            rng = MersenneTwister(21)

            # Act
            sampled = StrategyInterface.sample_action(strategy, 1, rng)

            # Assert
            @test sampled == 100
        end

        @testset "action_probability delegates through indexed local strategy" begin
            # Arrange
            strategy = IndexedStrategies.DenseVectorIndexedStrategy(
                StrategyInterface.StateContext,
                [
                    LocalStrategies.FiniteMixedStrategy((:x, :y), (0.2, 0.8)),
                    LocalStrategies.FiniteMixedStrategy((:u, :v), (0.55, 0.45)),
                ],
            )

            # Act
            p1 = StrategyInterface.action_probability(strategy, 1, :y)
            p2 = StrategyInterface.action_probability(strategy, 2, :u)

            # Assert
            @test p1 ≈ 0.8
            @test p2 ≈ 0.55
        end

        @testset "action_density delegates through indexed local strategy" begin
            # Arrange
            density_rule_1 = LocalStrategies.SamplerDensityStrategy(
                r -> 0.3,
                x -> x - 1,
                0:1,
            )
            density_rule_2 = LocalStrategies.SamplerDensityStrategy(
                r -> 0.4,
                x -> x / 2,
                0:1,
            )
            strategy = IndexedStrategies.DenseVectorIndexedStrategy(
                StrategyInterface.StateContext,
                [density_rule_1, density_rule_2],
            )

            # Act
            density = StrategyInterface.action_density(strategy, 2, 8.0)

            # Assert
            @test density == 4.0
        end

        @testset "local_strategy throws BoundsError for invalid integer context" begin
            # Arrange
            strategy = IndexedStrategies.DenseVectorIndexedStrategy(
                StrategyInterface.StateContext,
                [LocalStrategies.DeterministicStrategy(:x)],
            )

            # Act / Assert
            @test_throws BoundsError StrategyInterface.local_strategy(strategy, 0)
            @test_throws BoundsError StrategyInterface.local_strategy(strategy, 2)
        end
    end
end