"""
    _packedinds(A::AbstractRFP, i::Int, j::Int)

Returns the indices in the `data` field of an RFP array corresponding to `[i,j]`
"""
function _packedinds(A::AbstractRFP, i::Integer, j::Integer)
    n, k, l = _rfpsize(A)
    0 < i ≤ n && 0 < j ≤ n || throw(BoundsError("[$i, $j] is not in a square matrix of size $n"))
    neven = iseven(n)
    reverse4trans = A.transr == 'T' ? reverse : identity
    if A.uplo == 'L'
        i < j && return (0, 0)
        return reverse4trans(l < j ? (j - l, i + !neven - l) : (i + neven, j))
    end
    i > j && return (0, 0)
    return reverse4trans(l < (j + !neven) ? (i, j + !neven - l) : (l + neven + j, i))
end

function _parentsize(n::Integer)
    n > 0 || throw(ArgumentError("n = $n must be positive"))
    return n + iseven(n), (n + 1) >> 1
end

Base.parent(A::AbstractRFP) = A.data


function _rfpsize(A::AbstractRFP)
    dsz = size(A.data)
    k, l = A.transr == 'T' ? reverse(dsz) : dsz
    L = 2l
    isone(abs(k - L)) || throw(ArgumentError("size(A.data) = $dsz is not consistent with RFP"))
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
