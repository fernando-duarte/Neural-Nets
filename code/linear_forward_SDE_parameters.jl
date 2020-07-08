using DifferentialEquations, Plots, StatsPlots

# user inputs
n_draws = 100 # number observations to get normal
n_bins = 15 # number observations to get normal
datasize = 26 # number of datapoints along the path
tspan = (0.0f0,1.0f0) # range
true_A = Float32(-1.0) # value of A
u0 = hcat(true_A, 1.0) # initial positions

function trueSDEfunc(du,u,p,t)
    du[1] = u[1].*true_A .+ u[2].* p
    du[2] = 0
end

function trueNOISEfunc(du,u,p,t)
  du[1] = u[2]
end

t = range(tspan[1],tspan[2],length=datasize)

prob = SDEProblem(trueSDEfunc, trueNOISEfunc, u0,tspan, p = 1.0)
Y_T = [solve(prob, SOSRI(), saveat = t,).u[end,1][1] for i in 1:n_draws]
x = collect(0:1:99)
plot(x,Y_T, label="Distribution of Y_T")
histogram(Y_T,bins = n_bins)