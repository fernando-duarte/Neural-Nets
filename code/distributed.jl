using BenchmarkTools, CuArrays, GPUArrays, Plots
function threadded_map!(f::Function, A::Array, B::Array)
    Threads.@threads for i in 1:length(A)
        A[i] = f(B[i])
    end
  A
end
x, y = rand(10^7), rand(10^7)
kernel(y) = (y / 33f0) * (732.f0/y)
# on the cpu without threads:
single_t = @belapsed map!($kernel, $x, $y)

# "on the CPU with 4 threads (2 real cores):
thread_t = @belapsed threadded_map!($kernel, $x, $y)

# on the GPU:
xgpu, ygpu = cu(x), cu(y)
gpu_t = @belapsed begin
  map!($kernel, $xgpu, $ygpu)
  GPUArrays.synchronize($xgpu)
end
times = [single_t, thread_t, gpu_t]
speedup = maximum(times) ./ times
println("speedup: $speedup")
bar(["1 core", "2 cores", "gpu"], speedup, legend = false, fillcolor = :grey, ylabel = "speedup")