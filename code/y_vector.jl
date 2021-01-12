
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
import Pkg; Pkg.add("DifferentialEquations")
import Pkg; Pkg.build("Tracker")
import Pkg; Pkg.add("Tracker")

using Zygote, Statistics, Test, Tracker
using Flux, DiffEqFlux, DifferentialEquations
using Flux: @epochs
using Random
using IterTools: ncycle
using DiffEqSensitivity

### User inputs
T = 1.0f0
tspan = (0.0f0,T)
m=Float32[2.0]
v=Float32[1.0]
u0 = [1.0f0; 10.0f0] # initial guess [y,x]

nn = Chain(Dense(1,10, relu),Dense(10,2)) # neural net for Z_t
p1,re = Flux.destructure(nn)
ps = Flux.params(p1,u0) # or Flux.params(p3,u0) if you want to also optimize over p2

function drift_nn(u,p,t)
    [-(u[1] + re(p[1:length(p1)])([u0[1]])[1] + u[2]), 1.0f0] #[y,x] 
end

function stoch_nn(u,p,t)
    [re(p[1:length(p1)])([u0[1]])[2],1.0f0] #[y,x]
end

probSDE_nn = SDEProblem{false}(drift_nn,stoch_nn,u0,tspan,p1)

num_sim = 4 # number of paths to draw
function terminal_val_nn(u0,p)
    val = [Array(solve(probSDE_nn,EM(),dt=0.005,p=p,u0=u0,save_start=false,saveat=T,save_noise=false,sensealg=TrackerAdjoint()))[1] for i= 1:num_sim]
end

function loss_nn(m,v)
    paths = terminal_val_nn(u0,p1)
    loss = (mean(paths)-m[1])^2 + (var(paths)-v[1])^2
end

terminal_val_nn(u0, p1)

cb = function() #callback function to observe training
  display(loss_nn(m,v))
end
cb()
y0_pre = deepcopy(u0)
p1_pre = deepcopy(p3)
Z_pre = re(p1_pre)([u0[1]])[1]

# train model on the same data num_cycle times
num_cycles = 1
data = ncycle([(m, v)], num_cycles)

# train
num_sim = 50
opt = ADAM(0.1)
@epochs 5 Flux.train!(loss_nn, ps , data,  opt, cb=cb)

opt = ADAM(0.01)
@epochs 15 Flux.train!(loss_nn, ps , data,  opt, cb=cb)

num_sim = 100
opt = ADAM(0.001)
@epochs 30 Flux.train!(loss_nn, ps , data,  opt, cb=cb)

# check solution
@show y0_pre,Z_pre
@show u0,Z

num_sim = 1000
@show mean(terminal_val_nn(u0, p1)) 
@show var(terminal_val_nn(u0, p1)) 
