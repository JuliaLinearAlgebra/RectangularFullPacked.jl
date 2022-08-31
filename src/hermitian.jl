struct HermitianRFP{T<:BlasFloat} <: AbstractRFP{T}
    data::StridedMatrix{T}
    transr::Char
    uplo::Char
end

HermitianRFP(A::TriangularRFP) = HermitianRFP(A.data, A.transr, A.uplo)

function Base.getindex(A::HermitianRFP, i::Integer, j::Integer)
    ii, jj = _packedinds(A, Int(i), Int(j))
    return iszero(ii) ? conj(A[j, i]) : A.data[ii, jj]
end

function Ac_mul_A_RFP(A::Matrix{T}, uplo = :U) where {T<:BlasFloat}
    n = size(A, 2)
    par = Matrix{T}(undef, _parentsize(n))
    tr = T <: Complex ? 'C' : 'T'
    (ul = first(string(uplo))) ∈ "UL" || throw(ArgumentError("uplo must be either :L or :U"))
    return HermitianRFP(LAPACK_RFP.sfrk!('N', ul, tr, 1.0, A, 0.0, par), 'N', ul)
end

Base.copy(A::HermitianRFP) = HermitianRFP(copy(A.data), A.transr, A.uplo)
