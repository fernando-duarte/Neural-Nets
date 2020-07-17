#####################################################################
### Description: This file compares using the GPUs and not using the GPUs to solve a NN for an SDE. GPUs seem 4x faster
#####################################################################

import Pkg; Pkg.build("SpecialFunctions")
using DifferentialEquations, Plots, Flux, CuArrays
using Flux: @epochs
using Flux.Data: DataLoader

# user inputs
n_draws = 100 # number observations to get normal
datasize = 26 # number of datapoints along the path
tspan = (0.0f0,1.0f0) # range

function trueSDEfunc(du,u,p,t)
    du[1] = u[1].*(Float32(-1.0)) .+ u[2].*(Float32(-1.0))
    du[2] = 0
end

function trueNOISEfunc(du,u,p,t)
  du[1] = u[2]
end

function f(x)
    return x.^2
end

function g(x)
    return 2*x
end


t = range(tspan[1],tspan[2],length=datasize)
p =  1.0:0.01:3.0
p = collect(p)

function gen_data(parameter)
    prob = SDEProblem(trueSDEfunc, trueNOISEfunc, hcat(g(parameter), f(parameter)), tspan)
    Y_T = [solve(prob, SOSRI(), saveat = t,).u[end,1][1] for i in 1:n_draws]
end

data = [gen_data(p[i]) for i in 1:size(p,1)]

xdata = hcat(data...)
ydata = hcat(g(p),p)'
xdata_gpu = cu(xdata)
ydata_gpu = cu(ydata)

train_data = DataLoader(xdata, ydata, batchsize=50, shuffle=true)
train_data_gpu = DataLoader(xdata_gpu,ydata_gpu, batchsize=50,shuffle=true)

NN = Chain(x -> x,
             Dense(n_draws,20,relu),
             Dense(20,2)) 

NN_gpu = Chain(x -> x,
             Dense(n_draws,20,relu),
             Dense(20,2)) |> gpu

function loss(x,y) 
    pred = NN(x)
    totalloss = sum((pred .- y).^2)
    return totalloss
end

function loss_gpu(x,y) 
    pred = NN_gpu(x)
    totalloss = sum((pred .- y).^2)
    return totalloss
end

@show loss(xdata,ydata)
@show loss_gpu(xdata_gpu,ydata_gpu)
ps = Flux.params(NN)
ps_gpu = Flux.params(NN_gpu)
opt = ADAM(0.01)
@time @epochs 500 Flux.train!(loss, ps, train_data, opt) #takes 80 seconds
@show loss(xdata,ydata) 

@time @epochs 500 Flux.train!(loss_gpu,ps_gpu,train_data_gpu,opt) #takes 20 seconds
@show loss_gpu(xdata_gpu,ydata_gpu)



