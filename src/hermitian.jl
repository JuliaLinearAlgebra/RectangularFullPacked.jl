struct HermitianRFP{T<:BlasFloat} <: AbstractRFP{T}
    data::Matrix{T}
    transr::Char
    uplo::Char
end

# For a Hermitian matrix the diagonal elements must be real.
# Hence fhis conversion cannot be done without copying A.data.

#HermitianRFP(A::TriangularRFP) = HermitianRFP(A.data, A.transr, A.uplo)

function Hermitian(A::TriangularRFP{<:LinearAlgebra.BlasReal}, uplo::Symbol)
    Symbol(A.uplo) == uplo ||
        throw(ArgumentError("A.uplo = $(A.uplo) conflicts with argument uplo = $uplo"))
    return Hermitian(A)
end

function Hermitian(A::TriangularRFP{<:LinearAlgebra.BlasReal})
    return HermitianRFP(A.data, A.transr, A.uplo)
end

Base.copy(A::HermitianRFP{T}) where {T} = HermitianRFP{T}(copy(A.data), A.transr, A.uplo)

function Base.getindex(A::HermitianRFP, i::Integer, j::Integer)
    (A.uplo == 'L' ? i < j : i > j) && return conj(getindex(A, j, i))
    n, k, l = checkbounds(A, i, j)
    rs, doconj = _packedinds(A, Int(i), Int(j), iseven(n), l)
    val = A.data[first(rs), last(rs)]
    return doconj ? conj(val) : val
end

function Ac_mul_A_RFP(A::Matrix{T}, uplo = :U) where {T<:BlasFloat}
    n = size(A, 2)
    par = Matrix{T}(undef, _parentsize(n))
    tr = T <: Complex ? 'C' : 'T'
    ul = first(string(uplo))
    if ul ∉ "UL"
        throw(ArgumentError("uplo must be either :L or :U"))
    end
    return HermitianRFP(LAPACK_RFP.sfrk!('N', ul, tr, 1.0, A, 0.0, par), 'N', ul)
end

function syrk!(
    trans::AbstractChar,
    α::Real,
    A::StridedMatrix{T},
    β::Real,
    C::HermitianRFP{T},
) where {T}
    return HermitianRFP(
        LAPACK_RFP.sfrk!(C.transr, C.uplo, Char(trans), α, A, β, C.data),
        C.transr,
        C.uplo,
    )
end
