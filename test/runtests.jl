using Test, LinearAlgebra
import RectangularFullPacked: Ac_mul_A_RFP, TriangularRFP

@testset "Rectangular Full Pack Format" begin

    @testset "Core generic functionality: n = $n, uplo = $uplo, transr = $transr" for n in (6, 7), uplo in (:U, :L), transr in (:N, :T)

        A = rand(10, n)

        @testset "Hermitian" begin
            AcA_RFP = Ac_mul_A_RFP(A, uplo)

            @test size(AcA_RFP, 1) == n
            @test size(AcA_RFP, 2) == n
            @test size(AcA_RFP, 3) == 1
            @test_throws BoundsError AcA_RFP[0, 1]
            @test_throws BoundsError AcA_RFP[1, 0]
            @test_throws BoundsError AcA_RFP[n+1, 1]
            @test_throws BoundsError AcA_RFP[1, n+1]

            @test AcA_RFP[2, 1] == AcA_RFP[1, 2]
            @test AcA_RFP[end-2, end-1] == AcA_RFP[end-1, end-2]
        end

        @testset "Triangular" begin
            Atr_RFP = TriangularRFP(triu(A'A), uplo; transr)
            @test size(Atr_RFP, 1) == n
            @test size(Atr_RFP, 2) == n
            @test size(Atr_RFP, 3) == 1

            # Indexing not implemented yet for Atr_RFP
            @test_throws BoundsError Atr_RFP[0, 1]
            @test_throws BoundsError Atr_RFP[1, 0]
            @test_throws BoundsError Atr_RFP[n + 1, 1]
            @test_throws BoundsError Atr_RFP[1, n + 1]

            # @test Atr_RFP[2, 1] == Atr_RFP[1, 2]
            # @test Atr_RFP[end-2, end-1] == Atr_RFP[end-1, end-2]
        end
    end

    @testset "Hermitian with element type: $elty. Problem size: $n, uplo: $uplo" for elty in (
            Float32,
            Float64,
            Complex{Float32},
            Complex{Float64},
        ),
        n in (6, 7),
        uplo in (:L, :U),
        transr in (:N, elty <: Complex ? :C : :T)

        A = rand(elty, 10, n)
        AcA = A'A
        AcA_RFP = Ac_mul_A_RFP(A, uplo)
        o = ones(elty, n)

        @test AcA ≈ AcA_RFP
        @test AcA \ o ≈ AcA_RFP \ o
        @test inv(AcA) ≈ inv(AcA_RFP)
        @test inv(cholesky(AcA)) ≈ inv(factorize(AcA_RFP))
    end

    @testset "Triangular with element type: $elty. Problem size: $n, uplo: $uplo, transr: $transr" for elty in (
            Float32,
            Float64,
            Complex{Float32},
            Complex{Float64},
        ),
        n in (6, 7),
        uplo in (:L, :U),
        transr in (:N, elty <: Complex ? :C : :T)

        A = lu(rand(elty, n, n)).U
        A = uplo == :U ? A : copy(A')
        A_RFP = TriangularRFP(A, uplo; transr)
        o = ones(elty, n)

        @test A ≈ A_RFP
        @test A ≈ Array(A_RFP)
        @test A \ o ≈ A_RFP \ o
        @test inv(A) ≈ Array(inv(A_RFP))
    end
end
