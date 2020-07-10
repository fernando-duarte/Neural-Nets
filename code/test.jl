#####################################################################
### Project: Neural Nets
### Author: Charlie
### Date: 7/5/2020
### Description: This file tests a variety of NN and Stochastic DiffEQ functions to ensure everything is working properly
#####################################################################

## Loading Neceessary Packages

import Pkg; Pkg.add("Flux")
import Pkg; Pkg.add("DifferentialEquations")

using Random, Distributions, Flux, Plots, DifferentialEquations
using Flux: @epochs
using Flux.Data: DataLoader

#Functions used throughout

function loss(x,y) 
    pred = NN(x)
    totalloss = sum((pred .- y).^2)
    return totalloss
end

## Setting seed
Random.seed!(1234)

#1. NN Mean Var Estimation
    # Initial Inputs
    μ = -2.0:0.01:2.0 #mean
    n_draws = 300 #number of draws per mu, var pairing
    σ = 1.0:0.01:5.0

    σ = collect(σ)
    μ = collect(μ) # convert to an array

    # Drawing the dataset
    ydata = hcat(μ,σ)'
    ydata = collect(ydata)
    ydist = [Normal(μ[i], σ[i]) for i in 1:size(μ,1)]    
    xdata = [rand(Normal(ydata[1,i], ydata[2,i]),n_draws) for i in 1:size(μ,1)]
    xdata = hcat(xdata...)

    NN = Chain(x -> x,
                 Dense(n_draws,20,relu),
                 Dense(20,2))

    train_data = DataLoader(xdata, ydata, batchsize=50, shuffle=true)
    initial_loss = loss(xdata,ydata)
    ps = Flux.params(NN)
    opt = ADAM(0.1)
    @epochs 50 Flux.train!(loss, ps, train_data, opt)
    opt = ADAM(0.01)
    @epochs 50 Flux.train!(loss, ps, train_data, opt)
    final_loss = loss(xdata,ydata)
    return initial_loss, final_loss

# Erroring if a few conditions occur

if initial_loss == final_loss
    display("No Training Occurred")
end

if initial_loss/10 < final_loss
    display("Final loss very similar to initial loss")
end

if final_loss > 100.0
    display("Final loss too high after training")
end

#2. Mean Var Estimate with a function computing variance from the mean
    μ = -2.0:0.01:2.0 #mean
    n_draws = 400 #number of draws per mu, var pairing

    function f(x) #function you choose, where f(x) > 0 for all x
        x.^2
    end

    σ = f(μ)
    μ = collect(μ) # convert to an array

    # Drawing the dataset
    ydata = hcat(μ,σ)'
    ydata = collect(ydata)
    ydist = [Normal(μ[i], σ[i]) for i in 1:size(μ,1)]    
    xdata = [rand(Normal(ydata[1,i], ydata[2,i]),n_draws) for i in 1:size(μ,1)]
    xdata = hcat(xdata...)

    NN = Chain(x -> x,
                 Dense(n_draws,20,relu),
                 Dense(20,2))

    train_data = DataLoader(xdata, ydata, batchsize=50, shuffle=true)
    
    initial_loss = loss(xdata,ydata)
    ps = Flux.params(NN)
    opt = ADAM(0.1)
    @epochs 50 Flux.train!(loss, ps, train_data, opt)
    opt = ADAM(0.01)
    @epochs 50 Flux.train!(loss, ps, train_data, opt) 
    final_loss = loss(xdata,ydata)
    return initial_loss, final_loss

# Erroring if a few conditions occur

if initial_loss == final_loss
    display("No Training Occurred")
end

if initial_loss/10 < final_loss
    display("Final loss very similar to initial loss")
end

if final_loss > 100.0
    display("Final loss too high after training")
end

#3. Ploting Y_T for SDE solution to see if normal
# user inputs
n_draws = 500 # number observations to get normal
n_bins = 30 # number observations to get normal
datasize = 26 # number of datapoints along the path
tspan = (0.0f0,1.0f0) # range
true_A = Float32(-1.0) # value of A
true_B = Float32(-1.0) # value of B
u0 = hcat(true_A, 1.0) # initial positions

function trueSDEfunc(du,u,p,t)
    du[1] = u[1].*true_A .+ u[2].* true_B
    du[2] = 0
end

function trueNOISEfunc(du,u,p,t)
  du[1] = u[2]
end

t = range(tspan[1],tspan[2],length=datasize)

prob = SDEProblem(trueSDEfunc, trueNOISEfunc, u0,tspan)
Y_T = [solve(prob, SOSRI(), saveat = t).u[end,1][1] for i in 1:n_draws]
x = collect(0:1:99)
plot(x,Y_T, label="Distribution of Y_T")
histogram(Y_T,bins = n_bins)

## 4. NN to solve for initial parameters from the results of Y_T. 

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
    return x.^2
end

function g(x)
    return 2*x
end


t = range(tspan[1],tspan[2],length=datasize)
p =  1.0:0.01:3.0
p = collect(p)

function gen_data(parameter)
    prob = SDEProblem(trueSDEfunc, trueNOISEfunc, hcat(g(parameter), parameter),tspan, p = parameter)
    Y_T = [solve(prob, SOSRI(), saveat = t,).u[end,1][1] for i in 1:n_draws]
end

data = [gen_data(p[i]) for i in 1:size(p,1)]

xdata = hcat(data...)
ydata = hcat(g(p),p)'
train_data = DataLoader(xdata, ydata, batchsize=50, shuffle=true)


NN = Chain(x -> x,
             Dense(n_draws,20,relu),
             Dense(20,2))

function loss(x,y) 
    pred = NN(x)
    totalloss = sum((pred .- y).^2)
    return totalloss
end

initial_loss = loss(xdata,ydata)
ps = Flux.params(NN)
opt = ADAM(0.1)
@epochs 100 Flux.train!(loss, ps, train_data, opt)
opt = ADAM(0.01)
@epochs 100 Flux.train!(loss, ps, train_data, opt)
final_loss = loss(xdata,ydata)

return initial_loss, final_loss


if initial_loss == final_loss
    display("No Training Occurred")
end

if initial_loss/10 < final_loss
    display("Final loss very similar to initial loss")
end

if final_loss > 1000.0
    display("Final loss too high after training")
end