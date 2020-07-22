#include("build.jl")
using DiffEqSensitivity, StochasticDiffEq, Zygote, Statistics, Test
using Flux, DiffEqFlux
using IterTools: ncycle
using Flux: @epochs
using Random

#= Solve the linear BSDE
dy_t = -(y_t+Z_t+c)dt+Z_t dB with t ∈[0,T]
terminal condition: E[y(T)]=m and var[y(T)]=v
c, m and v are given parameters
We approximate Z_t with a neural-network
=#

T = 0.2f0
tspan = (0.0f0,T)
m=Float32[0.0]
v=Float32[0.1]
y0 = Float32[0.1] # initial guess

nn = Chain(Dense(1,10), Dense(10,1)) # neural net for Z_t
p1,re = Flux.destructure(nn)
p2 = Float32[1.0]  # constant c
p3 = [p1;p2]
ps = Flux.params(p1,y0) # or Flux.params(p3,y0) if you want to also optimize over p2

function drift_nn(y,p,t)
  dy = -(y[1]+re(p[1:length(p1)])(y)[1]+p[length(p1)+1:length(p1)+length(p2)][1])
  return [dy]
end
function stoch_nn(y,p,t)
  dy = re(p[1:length(p1)])(y)[1]
  return [dy]
end

probSDE_nn = SDEProblem(drift_nn,stoch_nn,y0,tspan,p3)

num_sim = 2 # number of paths to draw
function yT_nn(y0,p)
        [Array(solve(probSDE_nn,EM(),dt=0.005,p=p,u0=y0,save_start=false,saveat=T,save_noise=false,sensealg=TrackerAdjoint() ))[end] for j=1:num_sim]
end

loss_nn(m,v) = (mean(yT_nn(y0,p3))-m[1])^2 + (var(yT_nn(y0,p3))-v[1])^2

cb = function() #callback function to observe training
  display(loss_nn(m,v))
  cb() < 1e-1 && Flux.stop() # stop training if loss is small
end
cb()
y0_pre = deepcopy(y0)
p1_pre = deepcopy(p1)
Z_pre = re(p1_pre)(y0_pre)[1]

# train model on the same data num_cycle times
num_cycles = 1
data = ncycle([(m, v)], num_cycles)
# train
opt = ADAM(0.1)
@epochs 1 Flux.train!(loss_nn, ps , data,  opt, cb=cb)

num_sim = 256
@epochs 5 Flux.train!(loss_nn, ps , data,  ADAM(1e-5), cb=cb)

# check solution
true_y0 = exp(T)*m[1] + (exp(T)-1)*( sqrt(  2*v[1]/(1-exp(-2*T)) )+1 )
true_Z = sqrt(  2*v[1]/(1-exp(-2*T)) )
Z = re(p1)(y0)[1]

@show true_y0, true_Z
@show y0_pre,Z_pre
@show y0,Z
cb()

tol = 0.05
@testset "BSDE is solved" begin
    @test  loss_flux(m,v)<tol
    @test  abs(true_y0-y0[1])<tol
    @test  abs(true_Z-Z[1])<tol
end

# we can also compute gradient of y(T) or the loss function with respect to y0 and Z directly (and, e.g., build our own optimizer)
dyT_dy0,dyT_dp3 = Zygote.gradient((y0,p3)->Array(solve(probSDE_nn,EM(),dt=0.1,p=p3,u0=y0,saveat=T,sensealg=TrackerAdjoint() ))[end],y0,p3)
l(y0,p3) = (mean(yT_nn(y0,p3))-m[1])^2 + (var(yT_nn(y0,p3))-v[1])^2
dL_y0,dL_dp3 = Zygote.gradient((y0,p3)->l(y0,p3),y0,p3)
