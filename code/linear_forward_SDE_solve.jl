#####################################################################
### Project: Neural Nets
### Author: Charlie
### Date: 7/5/2020
### Description: This file creates solves a forward SDE
#####################################################################

using DifferentialEquations

# user inputs
datasize = 30
tspan = (0.0f0,1.5f0)
true_A = Float32(-1.0)
true_B = Float32(-1.0)
u0 = hcat(true_A, 0.0)

function trueSDEfunc(du,u,p,t)
    du[1] = u[1].*true_A .+ u[2].* true_B
    du[2] = 0
end

function trueNOISEfunc(du,u,p,t)
  du[1] = u[2]
end

t = range(tspan[1],tspan[2],length=datasize)

prob = SDEProblem(trueSDEfunc, trueNOISEfunc, u0,tspan)
solve(prob, SOSRI(), saveat = t)