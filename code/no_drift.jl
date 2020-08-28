
### Packages
import Pkg; Pkg.build("SpecialFunctions")
import Pkg; Pkg.build("SLEEFPirates")
import Pkg; Pkg.build("FFTW")
import Pkg; Pkg.add("Tracker")
import Pkg; Pkg.add("IterTools")
import Pkg; Pkg.build("DiffEqFlux")
import Pkg; Pkg.add("DiffEqFlux")
import Pkg; Pkg.build("Zygote")
import Pkg; Pkg.add("Zygote")
using Zygote, Statistics, Test, Tracker
using Flux, DiffEqFlux, DifferentialEquations
using Flux: @epochs
using Random
using IterTools: ncycle
using DiffEqSensitivity

### User inputs

T = 1.0f0
tspan = (0.0f0,T)
m=Float32[0.0]
v=Float32[5.0]
y0 = Float32[0.1] # initial guess

nn = Chain(Dense(1,10, relu),Dense(10,5, relu), Dense(5,1)) # neural net for Z_t
p1,re = Flux.destructure(nn)
p2 = Float32[1.0]  # constant c
p3 = [p1;p2]
ps = Flux.params(p1,y0) # or Flux.params(p3,y0) if you want to also optimize over p2

function drift_nn(y,p,t)
  dy = 0
  return [dy]
end

function stoch_nn(y,p,t)
  dy = re(p[1:length(p1)])(p2)[1]
  return [dy]
end

probSDE_nn = SDEProblem(drift_nn,stoch_nn,y0,tspan,p3)

num_sim = 2 # number of paths to draw
function yT_nn(y0,p)
        [Array(solve(probSDE_nn,EM(),dt=0.005,p=p,u0=y0,save_start=false,saveat=T,save_noise=false,sensealg=TrackerAdjoint() ))[end] for j=1:num_sim]
end

loss_nn(m,v) = (0.0*mean(yT_nn(y0,p1))-m[1])^2 + (var(yT_nn(y0,p1))-v[1])^2

yT_nn(y0, p1)
cb = function() #callback function to observe training
  display(loss_nn(m,v))
end
cb()
y0_pre = deepcopy(y0)
p1_pre = deepcopy(p1)
Z_pre = re(p1_pre)(y0_pre)[1]

# train model on the same data num_cycle times
num_cycles = 1
data = ncycle([(m, v)], num_cycles)

# train
num_sim = 50
opt = ADAM(1.0)
@epochs 1 Flux.train!(loss_nn, ps , data,  opt, cb=cb)

opt = ADAM(0.1)
@epochs 5 Flux.train!(loss_nn, ps , data,  opt, cb=cb)

num_sim = 100
opt = ADAM(0.01)
@epochs 10 Flux.train!(loss_nn, ps , data,  opt, cb=cb)

num_sim = 1000
opt = ADAM(0.001)
@epochs 10 Flux.train!(loss_nn, ps , data,  opt, cb=cb)

opt = ADAM(0.0001)
@epochs 10 Flux.train!(loss_nn, ps , data,  opt, cb=cb)

# check solution
true_y0 = exp(T)*m[1] + (exp(T)-1)*( sqrt(  2*v[1]/(1-exp(-2*T)) )+1 )
true_Z = sqrt(  2*v[1]/(1-exp(-2*T)) )
Z = re(p1)(y0)[1]

@show true_y0, true_Z
@show y0_pre,Z_pre
@show y0,Z