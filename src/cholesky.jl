struct CholeskyRFP{T<:BlasFloat} <: Factorization{T}
    data::StridedMatrix{T}
    transr::Char
    uplo::Char
end

LinearAlgebra.cholesky!(A::HermitianRFP{T}) where {T<:BlasFloat} =
    CholeskyRFP(LAPACK_RFP.pftrf!(A.transr, A.uplo, A.data), A.transr, A.uplo)
LinearAlgebra.cholesky(A::HermitianRFP{T}) where {T<:BlasFloat} = cholesky!(copy(A))
LinearAlgebra.factorize(A::HermitianRFP) = cholesky(A)

Base.copy(F::CholeskyRFP{T}) where {T} = CholeskyRFP{T}(copy(F.data), F.transr, F.uplo)

# Solve
(\)(A::CholeskyRFP, B::StridedVecOrMat) =
    LAPACK_RFP.pftrs!(A.transr, A.uplo, A.data, copy(B))
(\)(A::HermitianRFP, B::StridedVecOrMat) = cholesky(A) \ B

LinearAlgebra.inv!(A::CholeskyRFP) =
    HermitianRFP(LAPACK_RFP.pftri!(A.transr, A.uplo, A.data), A.transr, A.uplo)
LinearAlgebra.inv(A::CholeskyRFP) = LinearAlgebra.inv!(copy(A))
LinearAlgebra.inv(A::HermitianRFP) = LinearAlgebra.inv!(cholesky(A))

LinearAlgebra.parent(A::CholeskyRFP) = A.data
