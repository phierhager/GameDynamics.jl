using Test
using GameLab

const LF = GameLab.LearningSignals
const LD = GameLab.LearningDiagnostics

@testset "LearningDiagnostics" begin
    @testset "RunningStat default constructor and mean on empty" begin
        # Arrange
        s = LD.RunningStat()

        # Act
        μ = LD.mean_value(s)

        # Assert
        @test s.n == 0
        @test s.sum == 0.0
        @test μ == 0.0
    end

    @testset "RunningStat push! accumulates count and sum" begin
        # Arrange
        s = LD.RunningStat{Float64}()

        # Act
        LD.push!(s, 1.0)
        LD.push!(s, 2.5)
        LD.push!(s, -0.5)

        # Assert
        @test s.n == 3
        @test s.sum == 3.0
        @test LD.mean_value(s) == 1.0
    end

    @testset "RunningStat reset! clears state" begin
        # Arrange
        s = LD.RunningStat{Int}()
        LD.push!(s, 4)
        LD.push!(s, 6)

        # Act
        LD.reset!(s)

        # Assert
        @test s.n == 0
        @test s.sum == 0
        @test LD.mean_value(s) == 0.0
    end

    @testset "LearnerTrace default state" begin
        # Arrange
        tr = LD.LearnerTrace()

        # Act / Assert
        @test tr.t == 0
        @test tr.cumulative_utility == 0.0
        @test tr.best_fixed_utility == 0.0
        @test tr.cumulative_reward == 0.0
        @test LD.cumulative_regret(tr) == 0.0
        @test LD.average_utility(tr) == 0.0
        @test LD.average_reward(tr) == 0.0
    end

    @testset "LearnerTrace push! with bandit feedback updates utility only" begin
        # Arrange
        tr = LD.LearnerTrace{Float64}()
        fb = LF.BanditSignal(2, 1.25)

        # Act
        LD.push!(tr, fb)

        # Assert
        @test tr.t == 1
        @test tr.cumulative_utility == 1.25
        @test tr.cumulative_reward == 1.25
        @test tr.best_fixed_utility == 0.0
        @test LD.cumulative_regret(tr) == -1.25
    end

    @testset "LearnerTrace push! with full-information feedback updates regret benchmark" begin
        # Arrange
        tr = LD.LearnerTrace{Float64}()
        fb = LF.FullInformationSignal(2, 0.5, [0.1, 0.5, 0.9])

        # Act
        LD.push!(tr, fb)

        # Assert
        @test tr.t == 1
        @test tr.cumulative_utility == 0.5
        @test tr.cumulative_reward == 0.5
        @test tr.best_fixed_utility == 0.9
        @test LD.cumulative_regret(tr) == 0.4
        @test LD.average_utility(tr) == 0.5
        @test LD.average_reward(tr) == 0.5
    end

    @testset "LearnerTrace reset! clears all fields" begin
        # Arrange
        tr = LD.LearnerTrace{Float64}()
        LD.push!(tr, LF.BanditSignal(1, 2.0))
        LD.push!(tr, LF.FullInformationSignal(1, 1.0, [1.0, 3.0]))

        # Act
        LD.reset!(tr)

        # Assert
        @test tr.t == 0
        @test tr.cumulative_utility == 0.0
        @test tr.best_fixed_utility == 0.0
        @test tr.cumulative_reward == 0.0
    end

    @testset "utility_gap computes benchmark minus realized" begin
        # Arrange
        realized = 0.4
        benchmark = 1.1

        # Act
        gap = LD.utility_gap(realized, benchmark)

        # Assert
        @test gap ≈ 0.7
    end

    @testset "empirical_action_histogram! increments valid index" begin
        # Arrange
        counts = [0, 2, 1]

        # Act
        LD.empirical_action_histogram!(counts, 2)

        # Assert
        @test counts == [0, 3, 1]
    end

    @testset "empirical_action_histogram! throws on low out-of-bounds action" begin
        # Arrange
        counts = [1, 2, 3]

        # Act / Assert
        @test_throws BoundsError LD.empirical_action_histogram!(counts, 0)
    end

    @testset "empirical_action_histogram! throws on high out-of-bounds action" begin
        # Arrange
        counts = [1, 2, 3]

        # Act / Assert
        @test_throws BoundsError LD.empirical_action_histogram!(counts, 4)
    end

    @testset "empirical_action_frequencies! normalizes nonzero counts" begin
        # Arrange
        counts = [1, 1, 2]
        dest = zeros(Float64, 3)

        # Act
        LD.empirical_action_frequencies!(dest, counts)

        # Assert
        @test dest ≈ [0.25, 0.25, 0.5]
        @test sum(dest) ≈ 1.0
    end

    @testset "empirical_action_frequencies! returns uniform distribution for zero counts" begin
        # Arrange
        counts = [0, 0, 0, 0]
        dest = zeros(Float64, 4)

        # Act
        LD.empirical_action_frequencies!(dest, counts)

        # Assert
        @test dest ≈ fill(0.25, 4)
        @test sum(dest) ≈ 1.0
    end

    @testset "empirical_action_frequencies! throws on destination length mismatch" begin
        # Arrange
        counts = [1, 2, 3]
        dest = zeros(Float64, 2)

        # Act / Assert
        @test_throws ArgumentError LD.empirical_action_frequencies!(dest, counts)
    end
end