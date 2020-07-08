#####################################################################
### Project: Neural Nets
### Author: Charlie
### Date: 7/5/2020
### Description: This file creates a plots the distribution of the solution of the foward SDE where certain aspects are passed as parameters. 
#####################################################################

using DifferentialEquations, Plots

# user inputs
n_draws = 100 # number observations to get normal
n_bins = 15 # number observations to get normal
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
p = 1.0
prob = SDEProblem(trueSDEfunc, trueNOISEfunc, hcat(g(p), p),tspan, p = 1.0)
Y_T = [solve(prob, SOSRI(), saveat = t,).u[end,1][1] for i in 1:n_draws]
x = collect(0:1:99)
histogram(Y_T,bins = n_bins)