module LAPACK_RFP

using libblastrampoline_jll
using LinearAlgebra
using LinearAlgebra: BlasInt, checksquare, chkstride1, LAPACKException
using LinearAlgebra.BLAS: @blasfunc
using LinearAlgebra.LAPACK: chkdiag, chkside, chkuplo

liblapack_name = libblastrampoline_jll.libblastrampoline

# Rectangular full packed format

## Symmetric rank-k operation for matrix in RFP format.
for (f, elty, relty) in (
    (:dsfrk_, :Float64, :Float64),
    (:ssfrk_, :Float32, :Float32),
    (:zhfrk_, :ComplexF64, :Float64),
    (:chfrk_, :ComplexF32, :Float32),
)

    @eval begin
        function sfrk!(
            transr::Char,
            uplo::Char,
            trans::Char,
            alpha::Real,
            A::StridedMatrix{$elty},
            beta::Real,
            C::VecOrMat{$elty},
        )
            chkuplo(uplo)
            chkstride1(A)
            if trans in ('N', 'n')
                n, k = size(A)
            elseif trans in ('T', 't', 'C', 'c')
                k, n = size(A)
            end
            lda = max(1, stride(A, 2))

            ccall(
                (@blasfunc($f), liblapack_name),
                Cvoid,
                (
                    Ref{UInt8},
                    Ref{UInt8},
                    Ref{UInt8},
                    Ref{BlasInt},
                    Ref{BlasInt},
                    Ref{$relty},
                    Ptr{$elty},
                    Ref{BlasInt},
                    Ref{$relty},
                    Ptr{$elty},
                ),
                transr,
                uplo,
                trans,
                n,
                k,
                alpha,
                A,
                lda,
                beta,
                C,
            )
            C
        end
    end
end

# Cholesky factorization of a real symmetric positive definite matrix A
for (f, elty) in (
    (:dpftrf_, :Float64),
    (:spftrf_, :Float32),
    (:zpftrf_, :ComplexF64),
    (:cpftrf_, :ComplexF32),
)

    @eval begin
        function pftrf!(transr::Char, uplo::Char, A::VecOrMat{$elty})
            chkuplo(uplo)
            n = round(Int, div(sqrt(8length(A)), 2))
            info = Ref{BlasInt}()

            ccall(
                (@blasfunc($f), liblapack_name),
                Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}),
                transr,
                uplo,
                n,
                A,
                info,
            )
            A
        end
    end
end

# Computes the inverse of a (real) symmetric positive definite matrix A using the Cholesky factorization
for (f, elty) in (
    (:dpftri_, :Float64),
    (:spftri_, :Float32),
    (:zpftri_, :ComplexF64),
    (:cpftri_, :ComplexF32),
)

    @eval begin
        function pftri!(transr::Char, uplo::Char, A::VecOrMat{$elty})
            chkuplo(uplo)
            n = round(Int, div(sqrt(8length(A)), 2))
            info = Ref{BlasInt}()

            ccall(
                (@blasfunc($f), liblapack_name),
                Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}),
                transr,
                uplo,
                n,
                A,
                info,
            )

            A
        end
    end
end

# DPFTRS solves a system of linear equations A*X = B with a symmetric positive definite matrix A using the Cholesky factorization
for (f, elty) in (
    (:dpftrs_, :Float64),
    (:spftrs_, :Float32),
    (:zpftrs_, :ComplexF64),
    (:cpftrs_, :ComplexF32),
)

    @eval begin
        function pftrs!(
            transr::Char,
            uplo::Char,
            A::VecOrMat{$elty},
            B::StridedVecOrMat{$elty},
        )
            chkuplo(uplo)
            chkstride1(B)
            n = round(Int, div(sqrt(8length(A)), 2))
            if n != size(B, 1)
                throw(DimensionMismatch("B has first dimension $(size(B,1)) but needs $n"))
            end
            nhrs = size(B, 2)
            ldb = max(1, stride(B, 2))
            info = Ref{BlasInt}()

            ccall(
                (@blasfunc($f), liblapack_name),
                Cvoid,
                (
                    Ref{UInt8},
                    Ref{UInt8},
                    Ref{BlasInt},
                    Ref{BlasInt},
                    Ptr{$elty},
                    Ptr{$elty},
                    Ref{BlasInt},
                    Ref{BlasInt},
                ),
                transr,
                uplo,
                n,
                nhrs,
                A,
                B,
                ldb,
                info,
            )

            B
        end
    end
end

# Solves a matrix equation (one operand is a triangular matrix in RFP format)
for (f, elty) in (
    (:dtfsm_, :Float64),
    (:stfsm_, :Float32),
    (:ztfsm_, :ComplexF64),
    (:ctfsm_, :ComplexF32),
)

    @eval begin
        function tfsm!(
            transr::Char,
            side::Char,
            uplo::Char,
            trans::Char,
            diag::Char,
            alpha::$elty,
            A::Matrix{$elty},
            B::StridedVecOrMat{$elty},
        )
            chkuplo(uplo)
            chkside(side)
            chkdiag(diag)
            chkstride1(A)
            chkstride1(B)
            m, n = size(B, 1), size(B, 2)
            k, l = transr == 'N' ? size(A) : reverse(size(A))
            if k - (2l ≤ k) != m
                throw(
                    DimensionMismatch(
                        "First dimension of B must equal $(k - (2l ≤ k)), got $m",
                    ),
                )
            end
            ldb = max(1, stride(B, 2))

            ccall(
                (@blasfunc($f), liblapack_name),
                Cvoid,
                (
                    Ref{UInt8},
                    Ref{UInt8},
                    Ref{UInt8},
                    Ref{UInt8},
                    Ref{UInt8},
                    Ref{BlasInt},
                    Ref{BlasInt},
                    Ref{$elty},
                    Ptr{$elty},
                    Ptr{$elty},
                    Ref{BlasInt},
                ),
                transr,
                side,
                uplo,
                trans,
                diag,
                m,
                n,
                alpha,
                A,
                B,
                ldb,
            )

            return B
        end
    end
end

# Computes the inverse of a triangular matrix A stored in RFP format.
for (f, elty) in (
    (:dtftri_, :Float64),
    (:stftri_, :Float32),
    (:ztftri_, :ComplexF64),
    (:ctftri_, :ComplexF32),
)

    @eval begin
        function tftri!(transr::Char, uplo::Char, diag::Char, A::StridedMatrix{$elty})
            chkuplo(uplo)
            chkdiag(diag)
            chkstride1(A)
            k, l = transr == 'N' ? size(A) : reverse(size(A))
            n = k - (2l ≤ k)
            info = Ref{BlasInt}()

            ccall(
                (@blasfunc($f), liblapack_name),
                Cvoid,
                (
                    Ref{UInt8},
                    Ref{UInt8},
                    Ref{UInt8},
                    Ref{BlasInt},
                    Ptr{$elty},
                    Ref{BlasInt},
                ),
                transr,
                uplo,
                diag,
                n,
                A,
                info,
            )

            A
        end
    end
end

# Copies a triangular matrix from the rectangular full packed format (TF) to the standard full format (TR)
for (f, elty) in (
    (:dtfttr_, :Float64),
    (:stfttr_, :Float32),
    (:ztfttr_, :ComplexF64),
    (:ctfttr_, :ComplexF32),
)

    @eval begin
        function tfttr!(
            A::StridedMatrix{$elty},
            transr::Char,
            uplo::Char,
            Arf::VecOrMat{$elty},
        )
            chkuplo(uplo)
            info = Ref{BlasInt}()
            chkstride1(A)
            n = checksquare(A)
            trgt = (n * (n + 1)) >> 1
            length(Arf) == trgt || throw(
                DimensionMismatch(
                    "RFP storage for ($m, $n) array is $(length(Arf)) ≠ $trgt",
                ),
            )
            lda = max(1, stride(A, 2))

            ccall(
                (@blasfunc($f), liblapack_name),
                Cvoid,
                (
                    Ref{UInt8},
                    Ref{UInt8},
                    Ref{BlasInt},
                    Ptr{$elty},
                    Ptr{$elty},
                    Ref{BlasInt},
                    Ref{BlasInt},
                ),
                transr,
                uplo,
                n,
                Arf,
                A,
                lda,
                info,
            )

            A
        end
    end
end

# Copies a triangular matrix from the standard full format (TR) to the rectangular full packed format (TF).
for (f, elty) in (
    (:dtrttf_, :Float64),
    (:strttf_, :Float32),
    (:ztrttf_, :ComplexF64),
    (:ctrttf_, :ComplexF32),
)

    @eval begin
        function trttf!(
            Arf::VecOrMat{$elty},
            transr::Char,
            uplo::Char,
            A::StridedMatrix{$elty},
        )
            chkuplo(uplo)
            chkstride1(A)
            n = size(A, 1)
            lda = max(1, stride(A, 2))
            info = Ref{BlasInt}()
            Arflen = (n * (n + 1)) >> 1
            length(Arf) == Arflen ||
                throw(DimensionMismatch("length(Arf) = $(length(Arf)), should be $Arflen"))

            ccall(
                (@blasfunc($f), liblapack_name),
                Cvoid,
                (
                    Ref{UInt8},
                    Ref{UInt8},
                    Ref{BlasInt},
                    Ptr{$elty},
                    Ref{BlasInt},
                    Ptr{$elty},
                    Ref{BlasInt},
                ),
                transr,
                uplo,
                n,
                A,
                lda,
                Arf,
                info,
            )

            Arf
        end
    end
end

end
