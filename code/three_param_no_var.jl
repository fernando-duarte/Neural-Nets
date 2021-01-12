
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
import Pkg; Pkg.build("DifferentialEquations")
import Pkg; Pkg.add(Pkg.PackageSpec(name="DifferentialEquations", version="1.2.3"))
import Pkg; Pkg.build("ModelingToolkit")
import Pkg; Pkg.add("ModelingToolkit")
import Pkg; Pkg.build("CUDA")
import Pkg; Pkg.add("CUDA")
import Pkg; Pkg.build("ParameterizedFunctions")
import Pkg; Pkg.add("ParameterizedFunctions")
import Pkg; Pkg.build("Tracker")
import Pkg; Pkg.add("Tracker")
import Pkg; Pkg.add("CSV")
import Pkg; Pkg.add("Tables")
import Pkg; Pkg.add("DataFrames")

### Using Packages
using CSV, Tables, DataFrames
using Zygote, Statistics, Test, Tracker
using ModelingToolkit
using Flux, DiffEqFlux, DifferentialEquations
using Flux: @epochs
using Random
using IterTools: ncycle
using DiffEqSensitivity

### User inputs
T = 0.2f0
tspan = (0.0f0,T)
m=Float32[1.0]
v=Float32[0.0]
u0 = Float32[5.0] # initial guess
Z = Float32[5.0]
p2 = Float32[1.0]  # constant c
ps = Flux.params(u0,Z,p2) # or Flux.params(p1,u0) if you want to also optimize over all u0 parameters

function drift_nn(u,p,t)
    [-(u[1]+ p[2])]
end

function stoch_nn(u,p,t)
    [p[1] * 0.0f0]
end

probSDE_nn = SDEProblem{false}(drift_nn,stoch_nn,u0,tspan,ps)

num_sim = 6 # number of paths to draw
function yT_nn(u0,p)
    #sensealg=TrackerAdjoint()
    #sensealg = ReverseDiffAdjoint()
    [Array(solve(probSDE_nn,EM(),dt=0.005,p=p,u0=u0,save_start=false,saveat=T,save_noise=false,sensealg=TrackerAdjoint()))[end] for j=1:num_sim]
end

function loss_nn(m,v)
    paths = yT_nn(u0,[Z p2]) 
    loss = (mean(paths)-m[1])^2 + (var(paths)-v[1])^2
end

yT_nn(u0, [Z p2])

cb2 = function() #callback function to observe training
  display(loss_nn(m,v))
end
cb2()
Z_pre = deepcopy(Z)
p2_pre = deepcopy(p2)
u0_pre = deepcopy(u0)

# train model on the same data num_cycle times
num_cycles = 1
data = ncycle([(m, v)], num_cycles)

# train
num_sim = 100
opt = ADAM(1.0)
#opt = Momentum(0.01)
#opt = Nesterov(0.01)
@epochs 10 Flux.train!(loss_nn, ps , data,  opt, cb=cb2)

opt = ADAM(0.1)
@epochs 10 Flux.train!(loss_nn, ps , data,  opt, cb=cb2)

opt = ADAM(0.01)
@epochs 50 Flux.train!(loss_nn, ps , data,  opt, cb=cb2)

num_sim = 500
opt = ADAM(0.001)
@epochs 15 Flux.train!(loss_nn, ps , data,  opt, cb=cb2)

num_sim = 1000
opt = ADAM(0.0001)
@epochs 100 Flux.train!(loss_nn, ps , data,  opt, cb=cb2)

num_sim = 1000
opt = ADAM(0.00001)
@epochs 100 Flux.train!(loss_nn, ps , data,  opt, cb=cb2)

num_sim = 10000
opt = ADAM(0.00001)
@epochs 3000 Flux.train!(loss_nn, ps , data,  opt, cb=cb2)


u0_post = u0
Z_post = Z

@show u0_pre u0_post
@show Z_pre Z_post 


