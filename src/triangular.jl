struct TriangularRFP{T<:BlasFloat} <: AbstractRFP{T}
    data::Matrix{T}
    transr::Char
    uplo::Char
end

function TriangularRFP(A::Matrix{T}, uplo::Symbol = :U; transr::Symbol = :N) where {T}
    n = checksquare(A)
    ul = first(string(uplo))
    if ul ∉ "UL"
        throw(ArgumentError("uplo = $uplo should be :U or :L"))
    end
    tr = first(string(transr))
    if tr ∉ (T <: Complex ? "NC" : "NT")
        throw(ArgumentError("transr = $transr should be :N or :(T <: Complex ? :C : :T)"))
    end
    rfdims = _parentsize(n, tr ≠ 'N')
    return TriangularRFP(
        LAPACK_RFP.trttf!(similar(A, first(rfdims), last(rfdims)), tr, ul, A),
        tr,
        ul,
    )
end

function Base.Array(A::TriangularRFP{T}) where {T}
    n, k, l = _rfpsize(A)
    C = Array{T}(undef, (n, n))
    LAPACK_RFP.tfttr!(C, A.transr, A.uplo, A.data)
    return A.uplo == 'U' ? triu!(C) : tril!(C)
end

Base.copy(A::TriangularRFP) = TriangularRFP(copy(A.data), A.transr, A.uplo)

function Base.getindex(A::TriangularRFP{T}, i::Integer, j::Integer) where {T}
    n, k, l = checkbounds(A, i, j)
    (A.uplo == 'L' ? i < j : i > j) && return zero(T)
    rs, doconj = _packedinds(A, Int(i), Int(j), iseven(n), l)
    val = A.data[first(rs), last(rs)]
    return doconj ? conj(val) : val
end

function Base.setindex!(A::TriangularRFP{T}, x::T, i::Integer, j::Integer) where {T}
    n, k, l = checkbounds(A, i, j)
    if (A.uplo == 'L' ? i < j : i > j)
        if iszero(x)
            return x
        else
            throw(BoundsError("Attempt to assign a non-zero value to a systematic zero"))
        end
    end
    rs, doconj = _packedinds(A, Int(i), Int(j), iseven(n), l)
    return setindex!(A.data, doconj ? conj(x) : x, first(rs), last(rs))
end

LinearAlgebra.inv!(A::TriangularRFP) =
    TriangularRFP(LAPACK_RFP.tftri!(A.transr, A.uplo, 'N', A.data), A.transr, A.uplo)
LinearAlgebra.inv(A::TriangularRFP) = LinearAlgebra.inv!(copy(A))

LinearAlgebra.ldiv!(A::TriangularRFP{T}, B::StridedVecOrMat{T}) where {T} =
    LAPACK_RFP.tfsm!(A.transr, 'L', A.uplo, 'N', 'N', one(T), A.data, B)
function LinearAlgebra.ldiv!(
    A::Adjoint{T, TriangularRFP{T}},
    B::StridedVecOrMat{T}) where {T}
    Ap = A.parent
    tr = T <: Complex ? 'C' : 'T'
    return LAPACK_RFP.tfsm!(Ap.transr, 'L', Ap.uplo, tr, 'N', one(T), Ap.data, B)
end

LinearAlgebra.rdiv!(A::StridedVecOrMat{T}, B::TriangularRFP{T}) where {T} =
    LAPACK_RFP.tfsm!(B.transr, 'R', B.uplo, 'N', 'N', one(T), B.data, A)
function LinearAlgebra.rdiv!(
    A::StridedVecOrMat{T},
    B::Adjoint{T, TriangularRFP{T}}) where {T}
    Bp = B.parent
    tr = T <: Complex ? 'C' : 'T'
    return LAPACK_RFP.tfsm!(Bp.transr, 'R', Bp.uplo, tr, 'N', one(T), Bp.data, A)
end

(\)(A::TriangularRFP, B::StridedVecOrMat) = ldiv!(A, copy(B))
