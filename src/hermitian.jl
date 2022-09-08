struct HermitianRFP{T<:BlasFloat} <: AbstractRFP{T}
    data::Matrix{T}
    transr::Char
    uplo::Char
end

# For a Hermitian matrix the diagonal elements must be real.
# Hence fhis conversion cannot be done without copying A.data.

#HermitianRFP(A::TriangularRFP) = HermitianRFP(A.data, A.transr, A.uplo)

function Base.getindex(A::HermitianRFP, i::Integer, j::Integer)
    (A.uplo == 'L' ? i < j : i > j) && return conj(getindex(A, j, i))
    n, k, l = checkbounds(A, i, j)
    rs, doconj = _packedinds(A, Int(i), Int(j), iseven(n), l)
    val = A.data[first(rs), last(rs)]
    return doconj ?  conj(val) : val
end

function Ac_mul_A_RFP(A::Matrix{T}, uplo = :U) where {T<:BlasFloat}
    n = size(A, 2)
    par = Matrix{T}(undef, _parentsize(n))
    tr = T <: Complex ? 'C' : 'T'
    (ul = first(string(uplo))) ∈ "UL" || throw(ArgumentError("uplo must be either :L or :U"))
    return HermitianRFP(LAPACK_RFP.sfrk!('N', ul, tr, 1.0, A, 0.0, par), 'N', ul)
end

Base.copy(A::HermitianRFP) = HermitianRFP(copy(A.data), A.transr, A.uplo)
