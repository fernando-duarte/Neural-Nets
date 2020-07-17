#####################################################################
### Description: This file solves A\b for a sparse matrix 
### Results: Even for sparse matrices, the iterative solvers seems somewhat slower,
### though answers are roughly the same. Don't seem very compatible with the gpus. 
#####################################################################

### Packages
import Pkg; Pkg.add("SparseArrays")
using Test, BenchmarkTools, LinearAlgebra, Random, SparseArrays

### Setting Seed
Random.seed!(1234)

### user inputs
n = 10^7 #number of rows
m = 10^7 #number of columns

dv = rand(n)
ev = rand(m)

a = SymTridiagonal(dv, ev)
b = rand(n)

### Creating different solver functions

function standard_method(A, B)
    ans = A\B
    return collect(ans) # do this so CuArray is displayed as regular array
end

x = standard_method(a,b)

@test mean(a * x .≈ b) > .95