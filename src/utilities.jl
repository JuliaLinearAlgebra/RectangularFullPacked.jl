function checkbounds(A::AbstractRFP, i::Integer, j::Integer)
    n, k, l = _rfpsize(A)
    0 < i ≤ n && 0 < j ≤ n || throw(BoundsError("[$i, $j] isn't in an $n × $n matrix"))
    return n, k, l
end

"""
    _packedinds(A::AbstractRFP, i, j, neven, l)

Returns an index tuple into `A.data` where `ij` is in the correct index range and correct triangle

The triangle is checked before entry to this function because `ij` not being in the `A.uplo` triangle
is handled differently for triangular, which returns `zero(T)`, and Hermitian, which returns the
conjugate of `A[j, i]`.

`l`, the smaller dimension of the parent array, and `neven`, whether the virtual
size `n` is even, are already calculated and are passed to this function
"""
function _packedinds(A::AbstractRFP, i::Integer, j::Integer, neven::Bool, l::Int)
    return _packedinds(Int(i), Int(j), A.uplo == 'L', neven, A.transr == 'N', l)
end

function _packedinds(i::Int, j::Int, lower::Bool, neven::Bool, tr::Bool, l::Int)
    if lower
        conj = l < j
        inds = conj ? (j - l, i + !neven - l) : (i + neven, j)
    else
        conj = (j + !neven) ≤ l
        inds = conj ? (l + neven + j, i) : (i, j + !neven - l)
    end
    return tr ? (inds, conj) : (reverse(inds), !conj)
end

"""
    _parentsize(n::Integer)

Returns the size of the `data` field in an RFP array, representing a matrix of size `n`. `tr` is the value of `A.transr ≠ 'N'`.
"""
function _parentsize(n::Integer, tr::Bool = false)
    n > 0 || throw(ArgumentError("n = $n must be positive"))
    sz = (n + iseven(n), (n + 1) >> 1)
    return tr ? reverse(sz) : sz
end

"""
    _rfpsize(A::AbstractRFP)

Return a Tuple of `n`, the size of the square array, `k` the number of rows in the parent and `l` the number of columns

`k` and `l` are reversed when `A.transr` is not `'N'`.  That is, `k` should always be greater than `l`.

This function also checks the dimensions `k` and `l` for `isone(abs(2l - k))` which should always be `true`.
"""
function _rfpsize(A::AbstractRFP)
    dsz = size(A.data)
    k, l = A.transr == 'N' ? dsz : reverse(dsz)
    L = 2l
    isone(abs(k - L)) ||
        throw(ArgumentError("size(A.data) = $dsz is not consistent with RFP"))
    return k - (L < k), k, l
end

function Base.size(A::AbstractRFP, i::Integer)
    if i == 1 || i == 2
        n, k, l = _rfpsize(A)
        return n
    elseif i > 2
        return 1
    else
        return size(A.data, i)
    end
end
function Base.size(A::AbstractRFP)
    n, k, l = _rfpsize(A)
    return (n, n)
end

function LinearAlgebra.rmul!(A::AbstractRFP, B::Number)
    rmul!(A.data, B)
    return A
end

function LinearAlgebra.lmul!(A::Number, B::AbstractRFP)
    lmul!(A, B.data)
    return B
end
