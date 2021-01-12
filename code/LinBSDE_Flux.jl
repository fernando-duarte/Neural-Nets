include("build.jl")
using DifferentialEquations, DiffEqSensitivity, StochasticDiffEq, Zygote, Statistics, Test
using Flux
using Flux: @epochs
using IterTools: ncycle
using DifferentialEquations.EnsembleAnalysis
using Distributed

#= Solve the linear BSDE
dy_t = -(y_t+Z_t+1)dt+Z_t dB with t ∈[0,T]
terminal condition: E[y(T)]=m and var[y(T)]=v
m and v are given parameters
We know (and impose) Z_t = Z is constant
=#

T=0.3 # I get StackOverflowError for T=1.0 - don't know why
m=[0.0]
v=[0.5]

function drift(y,Z,t)
  dy = -(y[1]+Z[1]+1.0)
  return [dy]
end
function stoch(y,Z,t)
  dy = Z[1]
  return [dy]
end
y0 = [2.5]; Z = [3.0] # initial guess
true_y0 = [exp(T)*m[1] + (exp(T)-1)*( sqrt(  2*v[1]/(1-exp(-2*T)) )+1 )]; true_Z = [sqrt(  2*v[1]/(1-exp(-2*T)) )] # desired solution

probSDE = SDEProblem(drift,stoch,y0,(0.0,T),Z)
num_sim = 10 # number of paths to draw
function yT(y0,Z)
        [Array(solve(probSDE,EM(),dt=0.001,p=Z,u0=y0,save_start=false,saveat=T,save_noise=false,sensealg=TrackerAdjoint() ))[end] for j=1:num_sim]

end

loss(m,v) = (mean(yT(y0,Z))-m[1])^2 + (var(yT(y0,Z))-v[1])^2
cb = function() # callback function to observe training
    display(loss(m,v))
    #cb() < 1e-1 && Flux.stop() # stop training if loss is small
end
cb()

y0_pre = copy(y0); Z_pre = copy(Z) # remember initial y0, Z
ps = Flux.params(y0,Z) # optimize over y0, Z

# train model on the same data num_cycle times
num_cycles = 5
data = ncycle([(m, v)], num_cycles)

# train
@epochs 60 Flux.train!(loss, ps, data, ADAM(0.1), cb=cb)
@epochs 60 Flux.train!(loss, ps, data, ADAM(1e-5), cb=cb)

# compare to true solution
@show true_y0, true_Z
@show y0_pre,Z_pre
@show y0,Z
cb()

tol = 0.1
@testset "BSDE is solved" begin
    @test  loss(m,v)<tol
    @test  abs(true_y0[1]-y0[1])<tol
    @test  abs(true_Z[1]-Z[1])<tol
end

# we can also compute gradient of y(T) or the loss function with respect to y0 and Z directly (and, e.g., build our own optimizer)
dyT_dy0,dyT_dZ = Zygote.gradient((y0,Z)->Array(solve(probSDE,EM(),dt=0.1,u0=y0,p=Z,saveat=T,sensealg=TrackerAdjoint()))[end],y0,Z)
loss(y0,Z) = (mean(yT(y0,Z))-m[1])^2 + (var(yT(y0,Z))-v[1])^2
dL_y0,dL_dZ = Zygote.gradient((y0,Z)->loss(y0,Z),y0,Z)
