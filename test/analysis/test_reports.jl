using Test

const AnalysisReports = TestHarness.AnalysisReports

Base.@kwdef struct FlatIssue
    ok::Bool
    message::String
end

Base.@kwdef struct FlatReport
    valid::Bool
    issues::Vector{Any}
    family::Symbol = :flat_family
end

Base.@kwdef struct SectionIssue
    ok::Bool
    message::String
    code::Symbol = :issue
end

Base.@kwdef struct Section
    name::Symbol
    issues::Vector{Any}
end

Base.@kwdef struct SectionReport
    valid::Bool
    sections::Vector{Section}
    name::String = "section_report"
end

struct ValidOnly
    valid::Bool
end

struct MissingValid
    issues::Vector{Any}
end

@testset "analysis/reports.jl" begin
    @testset "is_valid_report reads the valid property" begin
        # Arrange
        rep = ValidOnly(true)

        # Act
        valid = AnalysisReports.is_valid_report(rep)

        # Assert
        @test valid === true
    end

    @testset "is_valid_report rejects objects without valid" begin
        # Arrange
        rep = MissingValid(Any[])

        # Act / Assert
        @test_throws ArgumentError AnalysisReports.is_valid_report(rep)
    end

    @testset "failure_messages filters only failing flat issues" begin
        # Arrange
        rep = FlatReport(
            valid = false,
            issues = Any[
                FlatIssue(ok = true, message = "all good"),
                FlatIssue(ok = false, message = "bad row"),
                123,
            ],
        )

        # Act
        msgs = AnalysisReports.failure_messages(rep)

        # Assert
        @test msgs == ["bad row", "123"]
        @test AnalysisReports.num_failures(rep) == 2
        @test AnalysisReports.first_failure_message(rep) == "bad row"
    end

    @testset "failure_messages formats sectioned issues with names and codes" begin
        # Arrange
        rep = SectionReport(
            valid = false,
            sections = [
                Section(
                    name = :shape,
                    issues = Any[
                        SectionIssue(ok = true, message = "ok", code = :pass),
                        SectionIssue(ok = false, message = "mismatch", code = :dim),
                    ],
                ),
                Section(
                    name = :probabilities,
                    issues = Any[
                        (ok = false, message = "negative entry", code = :neg),
                    ],
                ),
            ],
        )

        # Act
        msgs = AnalysisReports.failure_messages(rep)

        # Assert
        @test msgs == [
            "[shape] dim: mismatch",
            "[probabilities] neg: negative entry",
        ]
    end

    @testset "failure_messages rejects unsupported report shapes" begin
        # Arrange
        rep = ValidOnly(false)

        # Act / Assert
        @test_throws ArgumentError AnalysisReports.failure_messages(rep)
    end

    @testset "first_failure_message returns nothing when there are no failures" begin
        # Arrange
        rep = FlatReport(
            valid = true,
            issues = Any[FlatIssue(ok = true, message = "fine")],
        )

        # Act
        msg = AnalysisReports.first_failure_message(rep)

        # Assert
        @test isnothing(msg)
        @test AnalysisReports.num_failures(rep) == 0
    end

    @testset "pretty_validation_report includes family, validity, and failures" begin
        # Arrange
        rep = FlatReport(
            valid = false,
            issues = Any[
                FlatIssue(ok = false, message = "first"),
                FlatIssue(ok = false, message = "second"),
            ],
            family = :unit_validation,
        )

        # Act
        rendered = AnalysisReports.pretty_validation_report(rep)

        # Assert
        @test occursin("Validation report: unit_validation", rendered)
        @test occursin("valid = false", rendered)
        @test occursin("Failures:", rendered)
        @test occursin(" - first", rendered)
        @test occursin(" - second", rendered)
    end

    @testset "pretty_validation_report falls back to name and reports no failures" begin
        # Arrange
        rep = SectionReport(valid = true, sections = Section[], name = "named_report")

        # Act
        rendered = AnalysisReports.pretty_validation_report(rep)

        # Assert
        @test occursin("Validation report: named_report", rendered)
        @test occursin("valid = true", rendered)
        @test occursin("No failures.", rendered)
    end
end
