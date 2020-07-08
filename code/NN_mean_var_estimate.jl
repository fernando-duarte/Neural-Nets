#####################################################################
### Project: Neural Nets
### Author: Charlie
### Date: 7/5/2020
### Description: This file creates a NN that, given a dataset of draws from a normal random variable, estimates the mean and the variance. 
#####################################################################

using Random, Distributions, Flux, Plots
using Flux: @epochs

# Setting the seed
Random.seed!(123) 

# Initial Inputs
n_samples = 1 #number of mu, var combinations
μ = randn(1,n_samples) #mean
σ = randn(1,n_samples) #variance
σ = σ.^2
n_draws = 100 #number of draws per mu, var pairing

# Drawing the dataset
ydata = [μ;σ]
ydist = [Normal(μ[i], σ[i]) for i in 1:n_samples]
xdata = [rand(ydist[i],n_draws) for i in 1:n_samples]
xdata = xdata[1]
ydata = ydata[:,1]
data = [(xdata, ydata)]

NN = Chain(x -> x,
             Dense(n_draws,20,relu),
             Dense(20,2))

function loss(x,y) 
    pred = NN(x)
    totalloss = sum((pred .- y).^2)
    return totalloss
end

@show loss(xdata,ydata)
ps = Flux.params(NN)
evalcb() = @show(loss(xdata,ydata))
opt = ADAM(0.01)
@epochs 100 Flux.train!(loss, ps, data, opt)
@show(loss(xdata,ydata))
