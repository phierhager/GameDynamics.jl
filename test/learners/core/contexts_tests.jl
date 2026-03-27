using Test
using GameLab

const LC = GameLab.LearningContexts
const LI = GameLab.LearningInterfaces

@testset "LearningContexts" begin
    @testset "NullContext is a learning context" begin
        # Arrange
        ctx = LC.NullContext()

        # Act / Assert
        @test ctx isa LI.AbstractLearningContext
    end

    @testset "RoundContext stores round and payload" begin
        # Arrange
        ctx = LC.RoundContext(5, "meta")

        # Act / Assert
        @test ctx isa LI.AbstractLearningContext
        @test ctx.t == 5
        @test ctx.payload == "meta"
    end

    @testset "ObservationContext stores observation" begin
        # Arrange
        ctx = LC.ObservationContext((x = 1, y = 2))

        # Act / Assert
        @test ctx.observation == (x = 1, y = 2)
    end

    @testset "StateContext stores state" begin
        # Arrange
        ctx = LC.StateContext([1, 2, 3])

        # Act / Assert
        @test ctx.state == [1, 2, 3]
    end

    @testset "HistoryContext stores history" begin
        # Arrange
        history = [:a, :b, :c]
        ctx = LC.HistoryContext(history)

        # Act / Assert
        @test ctx.history === history
    end
end