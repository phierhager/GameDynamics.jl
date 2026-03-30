using Test
using Random
using GameLab

const LI = GameLab.LearningInterfaces
const LC = GameLab.LearningContexts
const LF = GameLab.LearningSignals
const FM = GameLab.FTPLLearners

@testset "FTPLLearners" begin
    @testset "FTPL constructor validates arguments" begin
        # Arrange / Act / Assert
        @test_throws ArgumentError FM.FTPL(0.0, 3)
        @test_throws ArgumentError FM.FTPL(0.1, 0)
    end

    @testset "state initializes zero cumulative utilities and scratch buffer" begin
        # Arrange
        learner = FM.FTPL(0.3, 4)

        # Act
        st = FM.FTPLState(learner)

        # Assert
        @test st.cumulative_utilities == zeros(4)
        @test st.scratch == zeros(4)
    end

    @testset "policy! throws because FTPL has no stable closed-form policy" begin
        # Arrange
        learner = FM.FTPL(0.2, 3)
        st = FM.FTPLState(learner)
        dest = zeros(3)

        # Act / Assert
        @test_throws ArgumentError LI.policy!(dest, learner, st, LC.NullContext())
    end

    @testset "act! returns valid action index and writes scratch values" begin
        # Arrange
        learner = FM.FTPL(0.5, 3)
        st = FM.FTPLState(learner)
        rng = MersenneTwister(12)

        # Act
        action = LI.act!(learner, st, LC.NullContext(), rng)

        # Assert
        @test 1 <= action <= 3
        @test length(st.scratch) == 3
    end

    @testset "update! accumulates full-information utility vector" begin
        # Arrange
        learner = FM.FTPL(0.5, 3)
        st = FM.FTPLState(learner)
        fb = LF.FullInformationSignal(2, 0.0, [1.0, -1.0, 0.5])

        # Act
        LI.update!(learner, st, fb)

        # Assert
        @test st.cumulative_utilities == [1.0, -1.0, 0.5]
    end

    @testset "update! throws on utility vector length mismatch" begin
        # Arrange
        learner = FM.FTPL(0.5, 3)
        st = FM.FTPLState(learner)
        fb = LF.FullInformationSignal(1, 0.0, [1.0, 2.0])

        # Act / Assert
        @test_throws ArgumentError LI.update!(learner, st, fb)
    end

    @testset "reset! restores zeroed state" begin
        # Arrange
        learner = FM.FTPL(0.5, 3)
        st = FM.FTPLState(learner)
        st.cumulative_utilities .= [1.0, 2.0, 3.0]
        st.scratch .= [4.0, 5.0, 6.0]

        # Act
        LI.reset!(learner, st)

        # Assert
        @test st.cumulative_utilities == zeros(3)
        @test st.scratch == zeros(3)
    end
end