# RectangularFullPacked

[![CI](https://github.com/JuliaLinearAlgebra/RectangularFullPacked.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/JuliaLinearAlgebra/RectangularFullPacked.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/JuliaLinearAlgebra/RectangularFullPacked.jl/branch/main/graph/badge.svg?token=440BGYEoar)](https://codecov.io/gh/JuliaLinearAlgebra/RectangularFullPacked.jl)

A Julia package for the Rectangular Full Packed (RFP) matrix storage format.

The RFP format stores a triangular or Symmetric/Hermitian matrix of size `n` in `(n * (n + 1))/2` storage locations consisting of three blocks from the original matrix.
The exact sizes and orientations of the blocks in the underlying `parent` array depend on whether the lower or upper triangle is stored and whether the parent array is transposed.
It also depends on whether the size of the matrix is even or odd as shown in Fig. 5 (p. 12) of [LAPACK Working Notes 199](https://netlib.org/lapack/lawnspdf/lawn199.pdf).

For example, starting with a 6 by 6 matrix whose elements are numbered 1 to 36 in column-major order
```julia
julia> using LinearAlgebra, RectangularFullPacked

julia> A = reshape(1:36, (6, 6))
6×6 reshape(::UnitRange{Int64}, 6, 6) with eltype Int64:
 1   7  13  19  25  31
 2   8  14  20  26  32
 3   9  15  21  27  33
 4  10  16  22  28  34
 5  11  17  23  29  35
 6  12  18  24  30  36
```
the lower triangular matrix `AL` is constructed by replacing the elements above the diagonal with zero.
```julia
julia> AL = tril!(collect(A))
6×6 Matrix{Int64}:
 1   0   0   0   0   0
 2   8   0   0   0   0
 3   9  15   0   0   0
 4  10  16  22   0   0
 5  11  17  23  29   0
 6  12  18  24  30  36
 ```
 `AL` requires the same amount of storage as does `A` even though there are only 21 potential non-zeros in `AL`.
 The RFP version of the lower triangular matrix
 ```julia
 julia> ArfpL = Int.(TriangularRFP(float.(A), :L))
6×6 Matrix{Int64}:
 1   0   0   0   0   0
 2   8   0   0   0   0
 3   9  15   0   0   0
 4  10  16  22   0   0
 5  11  17  23  29   0
 6  12  18  24  30  36
 ```
 provides the same displayed form but the underlying, "parent" array is 7 by 3
 ```julia
 julia> ALparent = Int.(ArfpL.data)
7×3 Matrix{Int64}:
 22  23  24
  1  29  30
  2   8  36
  3   9  15
  4  10  16
  5  11  17
  6  12  18
```

The three blocks of `AL` are the lower triangle of `AL[1:3, 1:3]`, stored as the lower triangle of `ALparent[2:4, :]`; the square block `AL[4:6, 1:3]` stored in `ALparent[5:7, :]`; and the lower triangle of `AL[4:6, 4:6]` stored transposed in `ALparent[1:3, :]`.

For odd values of n, the parent is of size `(n, div(n + 1, 2))` and the non-zeros in the first (n+1)/2 columns are in the same positions in `ALparent`.

For example,
```julia
julia> AL = tril!(collect(reshape(1:25, 5, 5)))
5×5 Matrix{Int64}:
 1   0   0   0   0
 2   7   0   0   0
 3   8  13   0   0
 4   9  14  19   0
 5  10  15  20  25

julia> ArfpL = Int.(TriangularRFP(float(AL), :L).data)
5×3 Matrix{Int64}:
 1  19  20
 2   7  25
 3   8  13
 4   9  14
 5  10  15
 ```

 RFP storage is especially useful for large positive definite Hermitian matrices because the Cholesky factor can be evaluated nearly as quickly (by applying Level-3 BLAS to the blocks) as in full storage mode but requiring about half the storage.

 A trivial example is
 ```julia
 julia> A = [2 1 2; 1 2 0; 1 0 2]
3×3 Matrix{Int64}:
 2  1  2
 1  2  0
 1  0  2

 julia> cholesky(Hermitian(A, :L))
Cholesky{Float64, Matrix{Float64}}
L factor:
3×3 LowerTriangular{Float64, Matrix{Float64}}:
 1.41421     ⋅         ⋅ 
 0.707107   1.22474    ⋅ 
 0.707107  -0.408248  1.1547

julia> ArfpL = HermitianRFP(TriangularRFP(float.(A), :L))
3×3 HermitianRFP{Float64}:
 2.0  1.0  1.0
 1.0  2.0  0.0
 1.0  0.0  2.0

 julia> cholesky!(ArfpL).data
3×2 Matrix{Float64}:
 1.41421    1.1547
 0.707107   1.22474
 0.707107  -0.408248
 ```
