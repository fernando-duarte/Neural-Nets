#####################################################################
### Description: This file solves A\b for a symmetric matrix 
### Results: Even with symmetric matrices, the iterative solvers seems somewhat slower,
### though answers are the same. Don't seem very compatible with the gpus. 
#####################################################################

### Packages
import Pkg; Pkg.add("CuArrays")
import Pkg; Pkg.add("IterativeSolvers")
import Pkg; Pkg.add("SparseArrays")
using CuArrays, Test, BenchmarkTools, IterativeSolvers, LinearAlgebra, Random, SparseArrays

### Setting Seed
Random.seed!(1234)

### Not allowing scalars for gpu runtime
CuArrays.allowscalar(false)

### user inputs
n = 10^4 #number of rows
m = 10^4 #number of columns

function droplower(A::SparseMatrixCSC)
    m,n = size(A)
    rows = rowvals(A)
    vals = nonzeros(A)
    V = Vector{eltype(A)}()
    I = Vector{Int}()
    J = Vector{Int}()
    for i=1:n
        for j in nzrange(A,i)
            rows[j]>i && break
            push!(I,rows[j])
            push!(J,i)
            push!(V,vals[j])
        end
    end
    return sparse(I,J,V,m,n)
end

### creating matrices
a = rand(n,m)
a = sparse(a)
a = droplower(a)
a = Symmetric(a)

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

### Calling Functions
@time standard_method(a,b)
@time standard_method(gpu_a,gpu_b)

@time cg_iteration(a,b)
#@time cg_iteration(gpu_a,b) #takes forever

@time minres_iteration(a,b) 
# @time minres_iteration(gpu_a,b) #takes forever

@time bicgstabl_iteration(a,b) # odd answer
bicgstabl_iteration(gpu_a,b) # odd answer

gmres_iteration(a,b)
gmres_iteration(gpu_a,b)
