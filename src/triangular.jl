struct TriangularRFP{T<:BlasFloat} <: AbstractRFP{T}
    data::StridedMatrix{T}
    transr::Char
    uplo::Char
end

function TriangularRFP(A::StridedMatrix, uplo::Symbol = :U)
    n = checksquare(A)
    Arf = similar(A, n + iseven(n), (n + 1) >> 1)
    if uplo == :U
        return TriangularRFP(LAPACK_RFP.trttf!(Arf, 'N', 'U', A), 'N', 'U')
    elseif uplo == :L
        return TriangularRFP(LAPACK_RFP.trttf!(Arf, 'N', 'L', A), 'N', 'L')
    else
        throw(ArgumentError("uplo must be either :U or :L but was :$uplo"))
    end
end
    
function Base.Array(A::TriangularRFP{T}) where {T}
    n, k, l = _rfpsize(A)
    C = Array{T}(undef, (n, n))
    LAPACK_RFP.tfttr!(C, A.transr, A.uplo, A.data)
    return A.uplo == 'U' ? triu!(C) : tril!(C)
end

Base.copy(A::TriangularRFP) = TriangularRFP(copy(A.data), A.transr, A.uplo)

function Base.getindex(A::TriangularRFP{T}, i::Integer, j::Integer) where {T}
    ii, jj = _packedinds(A, Int(i), Int(j))
    return iszero(ii) ? zero(T) : A.data[ii, jj]
end

function Base.setindex!(A::TriangularRFP{T}, x::T, i::Integer, j::Integer) where {T}
    ii, jj = _packedinds(A, Int(i), Int(j))
    if iszero(ii)
        if iszero(x)
            return x
        else
            throw(error("Attempt to assign a non-zero value to a systematic zero"))
        end
    end
    return setindex!(A.data, x, ii, jj)
end

LinearAlgebra.inv!(A::TriangularRFP) =
    TriangularRFP(LAPACK_RFP.tftri!(A.transr, A.uplo, 'N', A.data), A.transr, A.uplo)
LinearAlgebra.inv(A::TriangularRFP) = LinearAlgebra.inv!(copy(A))

ldiv!(A::TriangularRFP{T}, B::StridedVecOrMat{T}) where {T} =
    LAPACK_RFP.tfsm!(A.transr, 'L', A.uplo, 'N', 'N', one(T), A.data, B)
(\)(A::TriangularRFP, B::StridedVecOrMat) = ldiv!(A, copy(B))
