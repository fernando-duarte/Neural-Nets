#####################################################################
### Project: Neural Nets
### Author: Charlie
### Date: 7/5/2020
### Description: This file creates a NN that figures out what f and g are given ad ataset that consists of different realizations of Y_T for T = 1
#####################################################################

import Pkg; Pkg.add("DifferentialEquations")
import Pkg; Pkg.add("Flux")

using DifferentialEquations, Plots, Flux, Distributions, Random
using Flux: @epochs
using Flux.Data: DataLoader

# Setting seed
Random.seed!(123)
# user inputs
n_draws = 100 # number observations to get normal
datasize = 26 # number of datapoints along the path
tspan = (0.0f0,1.0f0) # range

function trueSDEfunc(du,u,p,t)
    du[1] = u[1].*(Float32(-1.0).+f(p)) .+ u[2].*p
    du[2] = 0
end

function trueNOISEfunc(du,u,p,t)
  du[1] = u[2]
end

function f(x)
    return 4*x.^2
end

function g(x)
    return 0.05*x
end

t = range(tspan[1],tspan[2],length=datasize)
p = 0.05:0.01:2.05
p = collect(p)

function gen_data(parameter)
    prob = SDEProblem(trueSDEfunc, trueNOISEfunc, hcat(g(parameter), parameter),tspan, p = parameter)
    Y_T = [solve(prob, SOSRI(), saveat = t,).u[end,1][1] for i in 1:n_draws]
end

data = [gen_data(p[i]) for i in 1:size(p,1)]

xdata = hcat(data...)
ydata = hcat(g(p),p)'

d = Binomial(1, 0.5)
ind = rand(d, size(p,1))
train_xdata = xdata[:, ind.==1]
train_ydata = ydata[:, ind.==1]
train_data = DataLoader(xdata[:, ind.==1], ydata[:, ind.==1], batchsize=50, shuffle=true)
test_xdata = xdata[:, ind.==0]
test_ydata = ydata[:, ind.==0]

NN = Chain(x -> x,
             Dense(n_draws,20,relu),
             Dense(20,2))

function loss(x,y) 
    pred = NN(x)
    totalloss = sum((pred .- y).^2)
    return totalloss
end

@show loss(train_xdata,train_ydata)
@show loss(test_xdata, test_ydata)
ps = Flux.params(NN)
opt = ADAM(0.1)
@time @epochs 500 Flux.train!(loss, ps, train_data, opt)
opt = ADAM(0.01)
@time @epochs 500 Flux.train!(loss, ps, train_data, opt)
@show loss(train_xdata,train_ydata)
@show loss(test_xdata,test_ydata)


