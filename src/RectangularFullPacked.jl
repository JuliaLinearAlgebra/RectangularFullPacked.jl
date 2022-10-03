module RectangularFullPacked

include("lapack.jl")

using LinearAlgebra
using LinearAlgebra: BlasFloat, checksquare

import Base: \
import LinearAlgebra.BLAS: syrk!

abstract type AbstractRFP{T} <: AbstractMatrix{T} end

include("utilities.jl")
include("triangular.jl")
include("hermitian.jl")
include("cholesky.jl")

export CholeskyRFP, 
    HermitianRFP,
    TriangularRFP

end
