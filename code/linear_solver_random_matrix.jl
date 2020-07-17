#####################################################################
### Description: This file solves A\b for a random matrix 
### Results: The iterative solvers do not seem to preform any better for a generic square matrix. 
### In fact, the produce less correct solutions and without any gain in computation speed
### Using gpu with a standard A\b method looks like the best way to do this, for large (10^4x10^4) matrices.
### For small matrices (10^2x10^2), don't bother with GPU. 
#####################################################################

### Packages
import Pkg; Pkg.add("CuArrays")
import Pkg; Pkg.add("IterativeSolvers")
using CuArrays, Test, BenchmarkTools, IterativeSolvers, LinearAlgebra, Random

### Setting Seed
Random.seed!(1234)

### Not allowing scalars for gpu runtime
CuArrays.allowscalar(false)

### user inputs
n = 10^4 #number of rows
m = 10^4 #number of columns

### creating matrices
a = rand(n,m)
b = rand(n,1)

gpu_a = cu(a)
gpu_b = cu(b)

### Creating different solver functions

function standard_method(A, B)
    ans = A\B
    return collect(ans) # do this so CuArray is displayed as regular array
end

function inverse_method(A, B)
    ans = inv(A)*B
    return collect(ans)
end

function cg_iteration(A,B) # best for symmetric positive semi-definite matrices 
    ans = cg(A,B)
    return collect(ans)
end

function minres_iteration(A,B) # best for symmetric positive indefinite matrices 
    ans = minres(A,B)
    return collect(ans)
end

function bicgstabl_iteration(A,B) # best for non-symmetric problems
    ans = bicgstabl(A,B)
    return collect(ans)
end

function idrs_iteration(A,B) # For non-symmetric strongly indefinite 
    ans = idrs(A,B)
    return collect(ans)
end

function gmres_iteration(A,B) # for non-symmetric with good precondition
    ans = gmres(A,B)
    return collect(ans)
end

## Calling functions
@time standard_method(a, b)
@time standard_method(gpu_a, gpu_b) # much faster 
@test mean(standard_method(a, b) .≈ standard_method(gpu_a, gpu_b)) > 0.95 #test should pass, but have to use approximate equality and allow for not all exactly equal

@time inverse_method(a, b) # slower than gpu method for previous one
#@time inverse_method(gpu_a, gpu_b) #inverse isn't defined for CuArray, try inv(gpu_a)
@test mean(standard_method(a, b) .≈ inverse_method(a, b)) > 0.95 # matches answers

@time cg_iteration(a,b) #bizarre answer, but runs
@time cg_iteration(gpu_a, b) # bizarre answer and really slow (doesn't work at all if not symmetric positive, semi-definite)

@time minres_iteration(a,b)
@time minres_iteration(gpu_a, b)
@test mean(minres_iteration(a,b) .≈ minres_iteration(gpu_a,b)) > 0.95 # somewhat dissimilar
@test mean(minres_iteration(a,b) .≈ standard_method(a,b)) > 0.95 # somewhat dissimilar

@time bicgstabl_iteration(a,b)
@time bicgstabl_iteration(gpu_a, b) 
@test mean(bicgstabl_iteration(a,b) .≈ bicgstabl_iteration(gpu_a,b)) > 0.95  # test fails
@test mean(standard_method(a, b) .≈ bicgstabl_iteration(a,b)) > 0.95  #test fails

@time gmres_iteration(a,b)
@time gmres_iteration(gpu_a, b)
@test mean(gmres_iteration(a,b) .≈ gmres_iteration(gpu_a,b)) > 0.95 #test fails
@test mean(standard_method(a, b) .≈ gmres_iteration(a,b)) > 0.95 # test fails
