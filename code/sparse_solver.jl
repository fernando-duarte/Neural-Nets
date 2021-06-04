import Pkg
Pkg.activate("joint_timing")
Pkg.instantiate()

using SparseArrays, LinearAlgebra, CUDA, BenchmarkTools, Test

n = 100
a = sprand(n,n,0.2) + sparse(I, n,n) 
a_cpu = Matrix(a)
a_gpu=CuArray(a_cpu)
A = CUSPARSE.CuSparseMatrixCSR(a) 
b = CUDA.rand(Float64, n) 
b_cpu = Array(b)
b_gpu = CuArray(b_cpu)
x = CuVector{Float64}(undef,n) 
tol = 1e-8 

@test Array(CUSOLVER.csrlsvqr!(A, b, x, tol, one(Cint),'O')) ≈ a_cpu\b_cpu
@btime CUSOLVER.csrlsvqr!($A, $b, $x, $tol, $one(Cint),$'O') #3.5 ms for n = 100,172s for n = 10000
@btime $a_cpu\$b_cpu #203 mus for n = 100, 11.1s for n = 10000
@btime $a_gpu\$b_gpu #2 ms for n = 100, 1.27s for n = 10000 

## Takeaways - both calculations produce the same result
## n = 100 cpu calculation is fastest, n = 10000 gpu is fastest, sparse arrays always slower
