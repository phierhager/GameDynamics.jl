using Test
using GameLab

const Encodings = GameLab.Encodings

@testset "analysis/encodings.jl" begin
    @testset "DenseEncoder encodes, reuses ids, and decodes" begin
        # Arrange
        enc = Encodings.DenseEncoder{String}()
        hinted = Encodings.sizehint!(enc, 4)

        # Act
        ida = Encodings.encode!(enc, "alpha")
        idb = Encodings.encode!(enc, "beta")
        ida_again = Encodings.encode!(enc, "alpha")

        # Assert
        @test hinted === enc
        @test ida == 1
        @test idb == 2
        @test ida_again == 1
        @test Encodings.has_encoding(enc, "alpha")
        @test Encodings.encode(enc, "beta") == 2
        @test Encodings.decode(enc, 1) == "alpha"
        @test Encodings.decode(enc, 2) == "beta"
    end

    @testset "DenseEncoder throws on unknown values and bad indices" begin
        # Arrange
        enc = Encodings.DenseEncoder{Symbol}()
        Encodings.encode!(enc, :seen)

        # Act / Assert
        @test_throws KeyError Encodings.encode(enc, :missing)
        @test_throws BoundsError Encodings.decode(enc, 0)
        @test_throws BoundsError Encodings.decode(enc, 2)
    end

    @testset "DenseEncoder reset! clears both directions" begin
        # Arrange
        enc = Encodings.DenseEncoder{Int}()
        Encodings.encode!(enc, 10)
        Encodings.encode!(enc, 20)

        # Act
        out = Encodings.reset!(enc)

        # Assert
        @test out === enc
        @test isempty(enc.to_id)
        @test isempty(enc.from_id)
        @test !Encodings.has_encoding(enc, 10)
        @test_throws KeyError Encodings.encode(enc, 10)
    end

    @testset "IdentityIntEncoder is a pure identity mapping" begin
        # Arrange
        enc = Encodings.IdentityIntEncoder()

        # Act
        encoded = Encodings.encode!(enc, 7)
        decoded = Encodings.decode(enc, 7)

        # Assert
        @test Encodings.has_encoding(enc, 1)
        @test !Encodings.has_encoding(enc, 0)
        @test !Encodings.has_encoding(enc, -3)
        @test encoded == 7
        @test Encodings.encode(enc, 11) == 11
        @test decoded == 7
        @test Encodings.sizehint!(enc, 10) === nothing
        @test Encodings.reset!(enc) === nothing
    end

    @testset "DenseIntRangeEncoder maps shifted dense ids" begin
        # Arrange
        enc = Encodings.DenseIntRangeEncoder(10, 3)

        # Act
        local_id = Encodings.encode!(enc, 12)
        external_id = Encodings.decode(enc, 2)

        # Assert
        @test Encodings.has_encoding(enc, 11)
        @test Encodings.has_encoding(enc, 13)
        @test !Encodings.has_encoding(enc, 10)
        @test !Encodings.has_encoding(enc, 14)
        @test local_id == 2
        @test Encodings.encode(enc, 11) == 1
        @test external_id == 12
        @test Encodings.reset!(enc) === enc
        @test Encodings.sizehint!(enc, 100) === nothing
    end

    @testset "DenseIntRangeEncoder rejects out-of-range values" begin
        # Arrange
        enc = Encodings.DenseIntRangeEncoder(-2, 2)

        # Act / Assert
        @test_throws KeyError Encodings.encode!(enc, -2)
        @test_throws KeyError Encodings.encode(enc, 1)
        @test_throws BoundsError Encodings.decode(enc, 0)
        @test_throws BoundsError Encodings.decode(enc, 3)
    end
end
