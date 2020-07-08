#####################################################################
### Project: Neural Nets
### Author: Charlie
### Date: 7/5/2020
### Description: This file creates a NN that, given a dataset of draws from a normal random variable, estimates the mean and the variance. 
#####################################################################

# Necessary Packages
using Random, Distributions, Flux, Plots
using Flux: @epochs
using Flux.Data: DataLoader

# Setting the seed
Random.seed!(123) 

# Initial Inputs
μ = -2.0:0.01:2.0 #mean
n_draws = 100 #number of draws per mu, var pairing

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

function loss(x,y) 
    pred = NN(x)
    totalloss = sum((pred .- y).^2)
    return totalloss
end

train_data = DataLoader(xdata, ydata, batchsize=50, shuffle=true)


@show loss(xdata,ydata)
ps = Flux.params(NN)
evalcb() = @show(loss(xdata,ydata))
opt = ADAM(0.01)
@epochs 500 Flux.train!(loss, ps, train_data, opt)
@show(loss(xdata,ydata))
