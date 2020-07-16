## load packages
import Pkg
#using Pkg

Pkg.add("FFTW")
# Pkg.build("FFTW")

Pkg.add("SpecialFunctions")
# Pkg.build("SpecialFunctions")

Pkg.add("GR")
# ENV["GRDIR"]="" ; Pkg.build("GR")

Pkg.add("PATHSolver")
#Pkg.build("PATHSolver")


using FFTW
using SpecialFunctions
using GR
using PATHSolver

Pkg.add("DataFrames")
Pkg.add("XLSX")
Pkg.add("Missings")
Pkg.add("NLsolve")
Pkg.add("Complementarity")
Pkg.add("NLopt")
Pkg.add("JuMP")


Pkg.add("ForwardDiff")
Pkg.add("GraphRecipes")
Pkg.add("LightGraphs")
Pkg.add("SimpleWeightedGraphs")
Pkg.add("Gadfly")

using Plots
using Gadfly
using GraphRecipes
using LightGraphs
using SimpleWeightedGraphs

using DataFrames, XLSX
using Missings
using LinearAlgebra, Random, Distributions, NLsolve, Complementarity
using Test
using NLopt
using JuMP

Pkg.add("Ipopt")
using Ipopt

using ForwardDiff
using Zygote
using CUDA

#= network primitives from the data
p_bar = total liabilities
c = outside assets
assets = total assets
w = net worth
b = outside liabilities
=#

## load data
xf = XLSX.readxlsx("node_stats_forsimulation_all.xlsx") 
data = vcat( [(XLSX.eachtablerow(xf[s]) |> DataFrames.DataFrame) for s in XLSX.sheetnames(xf)]... )
unique!(data) # delete duplicate rows


data = data[isless.(0,data.w), :]
data = data[isequal.(data.qt_dt,192), :] # keep quarter == 195 = 2008q4
sort!(data, :assets, rev = true)
data = data[1:10,:] # keep small number of nodes, for testing
N = size(data,1) # number of nodes

# rescale units
units = 1e6;
data[:,[:w, :c, :assets, :p_bar, :b]] .= data[!,[:w, :c, :assets, :p_bar, :b]]./units

# fake missing data to make problem feasible
data.b[:] .= missing
data.c[:] .= missing

# create network variables
data.f = data.p_bar .- data.b # inside liabilities
data.d =  data.assets .- data.c # inside assets

# keep track of missing variables
col_with_miss = names(data)[[any(ismissing.(col)) for col = eachcol(data)]] # columns with at least one missing 
#data_nm = dropmissing(data, disallowmissing=true) # drop missing
data_nm = coalesce.(data, data.w .+ 0.01 .+ rand(N)) # replace missing by a value
nm_c = findall(x->x==0,ismissing.(data.c))
nm_b = findall(x->x==0,ismissing.(data.b))

# take a look
names(data) # column names
describe(data)
show(data, true)

# parameters
g0 = 0.0 # bankruptcy cost

# function shock(c)
#     # draw shocks that produce the probability of default δ 
#     a = 1
#     b = log.(data.delta)./(log.(1 .-data.w./c))
#     dist = Beta.(a,[b...])
#     draws = rand.(dist, 1)
#     vcat(draws...).*c
#     #rand(Float64,N) # normal
# end

## Optimization

# initial values
rng = MersenneTwister(1234);
dist = Beta.(2,fill(2,N,1))
x0 = rand.(dist, 1)
x0 = fill(0.0,N,1)
A0 = rand(rng,N,N);[A0[i,i]=0.0 for i=1:N];A0=LowerTriangular(A0);

# set up optimization


#m = Model(Ipopt.Optimizer) # settings for the solver
#m = optimizer_with_attributes(Ipopt.Optimizer, "start_with_resto" => "yes", "linear_solver"=>"mumps")
m = Model()
m = Model(with_optimizer(Ipopt.Optimizer, start_with_resto="yes", linear_solver="mumps", max_iter=100000))

@variable(m, 0.0<=p[i=1:N]<=data.p_bar[i], start = data.p_bar[i]) 
@variable(m, 0.0<=c[i=1:N]<=data.assets[i], start = data_nm.c[i])  
@variable(m, 0.0<=b[i=1:N]<=data.p_bar[i], start = data_nm.b[i])   
@variable(m, 0.0<=A[i=1:N, j=1:N] <= 1.0 , start=A0[i,j])  # start=A0[i,j]

@constraint(m, sum(A,dims=2).*data.p_bar .== data_nm.p_bar .- b) # payments to other nodes add up to inside liabilities f
@constraint(m, A' * data.p_bar .== data.assets .- c ) # payments from other nodes add up to inside assets d

# liabilities are net liabilities: A[i,i]=0 and A[i,j]A[j,i]=0
@constraint(m, [i = 1:N], A[i,i]==0.0)
for i=1:N
    j=1
    while j < i
        @complements(m, 0.0 <= A[i,j],  A[j,i] >= 0.0)
        j += 1
    end
end

# register max and min non-linear functions
maxfun(x) = max(x, 0.0)
for i=1:N
    str = "minfun$i(x) = min(x, data_nm.p_bar[$i])"
	ex = Meta.parse(str)
	eval(ex)
    
    str = "∇minfun$i(x) = x<=data_nm.p_bar[$i] ? 1.0 : 0.0"
    ex = Meta.parse(str)
	eval(ex)
    
    str = "∇²minfun$i(x) = 0.0"
    ex = Meta.parse(str)
	eval(ex)
    
    str = "JuMP.register(m,:minfun$i, 1, minfun$i, ∇minfun$i, ∇²minfun$i)"
    ex = Meta.parse(str)
	eval(ex)
end

# JuMP.register(m, :maxfun, 2, maxfun, autodiff=true)
#JuMP.register(m, :minfun, 2, minfun, autodiff=true)

∇maxfun(x) = x<=0.0 ? 0.0 : 1.0
∇²maxfun(x) = 0.0
JuMP.register(m,:maxfun, 1, maxfun, ∇maxfun, ∇²maxfun)

# clearing vector
times(x,y) = x*y
JuMP.register(m,:times, 2, times, autodiff = true)

myexpr = []
for i=1:N
    push!(myexpr, (1+g0)*(sum(times(A[j,i],p[j]) for j in 1:N) + c[i] - x0[i]*c[i]) - g0*data_nm.p_bar[i])
end
# myexpr = (1+g0)*(sum(A[j,:]*p[j] for j in 1:N) .+ c .- x0.*c) .- g0*data_nm.p_bar[i]
@variable(m, aux[i=1:N])
@constraint(m, aux .== myexpr )
for i = 1:N
    #@NLconstraint(m,  minfun(data_nm.p_bar[i], maxfun(aux[i],0)) == p[i] ) 
#    @NLconstraint(m,  minfun(maxfun(aux[i],0),data_nm.p_bar[i]) == p[i] ) 
    str = "@NLconstraint(m,  minfun$i(maxfun(aux[$i])) == p[$i] )"
    ex = Meta.parse(str)
	eval(ex)
end

#[fix(c[i], data.c[i]; force=true) for i  in nm_c]
#[fix(b[i], data.b[i]; force=true) for i  in nm_b]

@NLobjective(m, Min , sum(-p[i] for i=1:N) ) #*sum(x[i]+p_bar[i]-p[i] for i=1:N) 

JuMP.optimize!(m)
termination_status(m)
objective_value(m)

psol = JuMP.value.(p)
csol = JuMP.value.(c)
bsol = JuMP.value.(b)
Asol = JuMP.value.(A)
auxsol = JuMP.value.(aux)

tol = 1e-6
@testset "check solution" begin

    @test all(0 .<= psol.<= data_nm.p_bar)    
    @test all(0 .<= csol.<= data_nm.assets)
    @test all(0 .<=Asol.<=1)
    
    @test norm( sum(Asol,dims=2).* data_nm.p_bar .- (data_nm.p_bar .- bsol))  < tol
    @test norm( Asol' * data_nm.p_bar .- (data.assets .- csol)) < tol

    @test norm(diag(Asol)) < tol
    @test norm([Asol[i,j]*Asol[j,i] for i=1:N , j=1:N]) < tol
    
    @test norm( psol .- min.(data.p_bar, max.((1+g0)*(Asol'*psol .+ csol .- x0 .* csol) .-g0.*data.p_bar,0.0)) ) < tol
    
end


Pkg.add("NLPModelsJuMP")
Pkg.add("NLPModels")
Pkg.add("MathOptInterface")
Pkg.add("OptimizationProblems")
Pkg.add("JSOSolvers")
Pkg.add("NLPModelsIpopt")

using NLPModelsJuMP
using NLPModels
using MathOptInterface
using OptimizationProblems
using JSOSolvers
using NLPModelsIpopt
Pkg.add(Pkg.PackageSpec(url="https://github.com/fernando-duarte/Percival.jl"))
using Percival

nlp=MathOptNLPModel(m)

stats = ipopt(nlp)
print(stats)
stats.solver_specific[:internal_msg]

psol2 = stats.solution[1:N]
csol2 = stats.solution[N+1:2*N]
bsol2 = stats.solution[2*N+1:3*N]
Asol2 = reshape(stats.solution[3*N+1:end-10],N,N)
auxsol2 = stats.solution[end-9:end]

tol = 1e-6
@testset "check solution 2" begin

    @test all(0 .<= psol2.<= data_nm.p_bar)    
    @test all(0 .<= csol2.<= data_nm.assets)
    @test all(0 .<=Asol2.<=1)
    
    @test norm( sum(Asol2,dims=2).* data_nm.p_bar .- (data_nm.p_bar .- bsol2))  < tol
    @test norm( Asol2' * data_nm.p_bar .- (data.assets .- csol2)) < tol

    @test norm(diag(Asol2)) < tol
    @test norm([Asol2[i,j]*Asol2[j,i] for i=1:N , j=1:N]) < tol
    
    @test norm( psol2 .- min.(data.p_bar, max.((1+g0)*(Asol2'*psol2 .+ csol2 .- x0 .* csol2) .-g0.*data.p_bar,0.0)) ) < tol
    
end

percival(nlp)


# clearing vector
contraction_iter(x0,20,c)
function contraction(p,x,c)
        minfun.(data_nm.p_bar, maxfun.(aux,0)) 
end
contraction_iter(x, n ,c) = n <= 0 ? data_nm.p_bar  : contraction(contraction_iter(x,n-1,c),x,c)


function test(q,vv...)
	vv
end
test(1.0,JuMP.all_variables(m)...)


myexpr = []
for i=1:N
    push!(myexpr, (1-g0)*(sum(a[j,i]*p[j] for j in 1:N) +c[i]+x ) + g0*data_nm.p_bar[i] )
end

@variable(m, aux[i=1:N])
for i = 1:N
	@constraint(m, aux[i] .== myexpr[i] )
end




for i = 1:N
    for j=1:D
        @NLconstraint(m,  minfun(data.p_bar[i], maxfun(aux[i,j],0)) == p[i,j] ) 
    end
end

[fix(c[i], data.c[i]; force=true) for i  in nm_c]
[fix(b[i], data.b[i]; force=true) for i  in nm_b]

@NLobjective(m, Max , sum(sum( x0[i,j]*c[i]+data.p_bar[i]-p[i,j] for i=1:N)/D for j=1:D) ) #*sum(x[i]+p_bar[i]-p[i] for i=1:N) 



x0 = rand(10)
pinf = nlsolve(p->contraction(p, x0,data_nm.c)-p, data_nm.p_bar, autodiff = :forward)
@test norm(contraction_iter(x0,100,data_nm.c)-pinf.zero)<tol

 -sum(contraction(data_nm.p_bar,x0,c)
	
@NLobjective(m, Max , sum(sum( x0[i,j]*c[i]+data.p_bar[i]-p[i,j] for i=1:N)/D for j=1:D) ) #*sum(x[i]+p_bar[i]-p[i] for i=1:N) 

#unset_silent(m)
JuMP.optimize!(m)

termination_status(m)
objective_value(m)


Gadfly.spy(Asol)

# Pkg.add("JLD")
# using JLD
# save("/home/ec2-user/SageMaker/Test-AWS/net_opt.jld", "Asol", Asol,"data",data)

## plot
Aplot = deepcopy(Asol)
Aplot[Aplot.<1e-3] .=0

Aplot2 = deepcopy(Asol2)
Aplot2[Aplot2.<1e-3] .=0
    
# attributes here: https://docs.juliaplots.org/latest/generated/graph_attributes/
#method `:spectral`, `:sfdp`, `:circular`, `:shell`, `:stress`, `:spring`, `:tree`, `:buchheim`, `:arcdiagram` or `:chorddiagram`.


graphplot(LightGraphs.DiGraph(Aplot),
          nodeshape=:circle,
          markersize = 0.05,
          node_weights = data.assets,
          markercolor = range(colorant"yellow", stop=colorant"red", length=N),
          names = data.nm_short,
          fontsize = 8,
          linecolor = :darkgrey,
          edgewidth = (s,d,w)->500*Asol[s,d],
          arrow=true,
          method= :circular, #:chorddiagram,:circular,:shell
          )



## fixed point
function contraction(p,x,c)
        min.(data_nm.p_bar, max.((1+g0)*(A0'*p .+ c .- x.*c) .- g0.*data_nm.p_bar,0)) 
end
contraction_iter(x, n::Integer,c) = n <= 0 ? data_nm.p_bar  : contraction(contraction_iter(x,n-1,c),x,c)

x0 = [0.88,0.92,0.2]

pinf = nlsolve(p->contraction(p, x0,data_nm.c)-p, data_nm.p_bar, autodiff = :forward)
@test norm(contraction_iter(x0,100,data_nm.c)-pinf.zero)<1e-6

contraction_iter(x0,200,data_nm.c)
	
Zygote.gradient(c -> -sum(contraction(data_nm.p_bar,x0,c)),data_nm.c)
ForwardDiff.gradient(c -> -sum(contraction(data_nm.p_bar,x0,c)),data_nm.c)

Zygote.gradient(c -> -sum(contraction_iter(x0,20,c)),data_nm.c)
ForwardDiff.gradient(c -> -sum(contraction_iter(x0,20,c)),data_nm.c)

pdf_beta(x,a,b) = x.^(a.-1.0).*(1.0 .-x).^(b.-1.0)
b_data(c) = log.(data_nm.delta)./(log.(1 .- data_nm.w./c))
a_data = 1.0
real.(Zygote.gradient(c-> pdf_beta(0.1,a_data,b_data(c))[2],data_nm.c))
imag.(Zygote.gradient(c-> pdf_beta(0.1,a_data,b_data(c))[2],data_nm.c))


a = 1.0
beta_b(c) = log.(data_nm.delta)./(log.(1 .- data_nm.w./c))
dist = Beta.(a,[b...])

betapdf(t,a,b) = t.^(a.-1.0).*(1.0 .-t).^(b.-1.0)


Zygote.gradient(c -> -sum(contraction_iter(x0,20,c).*betapdf(x0,a,beta_b(c))),data_nm.c)



tt = 0.332
aaa=a
bbb=1.3
Zygote.gradient(t->betapdf(t,a,bbb),tt)
(1-tt)^(bbb-2)*tt^(aaa-2)*(aaa+2*tt-aaa*tt-bbb*tt-1)

Zygote.gradient(a->betapdf(tt,a,bbb),aaa)
tt^(aaa-1)*log(tt)*(1-tt)^(bbb-1)

Zygote.gradient(c->betapdf(tt,aaa,beta_b(c))[1],1.0)



histogram(pdf.(dist,rand(10)))

shock() = vcat(rand.(dist, 1)...)
x0 = shock()
Zygote.gradient(c -> -sum(contraction_iter(x0,4,c)), data_nm.c)
function loss(c)
    x0 = shock()
    sum(contraction_iter(x0,4,c))
end
Zygote.gradient(c -> loss(c), data_nm.c)
check DistributionsAD.jl, Turing.jl
Flux.train!

Use https://github.com/ajt60gaibb/FastGaussQuadrature.jl 319 to get the quadrature rates, use a CUArray and broadcast your function across the array, and then accumulate according to the quadrature weights.
	
	https://github.com/giordano/Cuba.jl
	
ApproxFun.jl, you can do F = cumsum(Fun(f, 0..a)), and then evaluate F(x) very quickly — this works by first constructing a polynomial approximation of f(x) on [0,a] and then forming the polynomial F(x) that is the indefinite integral.
		
		Pkg.add("SparseGrids")

		
# Test GPU movement inside the call to `gradient`
@testset "GPU movement" begin
  r = rand(Float32, 3,3)
  @test gradient(x -> sum(cu(x)), r)[1] isa AbstractArray
end

@testset "basic bcasting" begin
  a = cu(Float32.(1:9))
  v(x, n) = x .^ n
  pow_grada = cu(Float32[7.0, 448.0, 5103.0, 28672.0, 109375.0, 326592.0, 823543.0, 1.835008e6, 3.720087e6])
  @test gradient(x -> v(x, 7) |> sum, a) == (pow_grada,)
  w(x) = broadcast(log, x)
  log_grada = cu(Float32[1.0, 0.5, 0.33333334, 0.25, 0.2, 0.16666667, 0.14285715, 0.125, 0.11111111])
  @test gradient(x -> w(x) |> sum, a) == (log_grada,)
end


# Zygote.hessian(c1 -> -sum(contraction_iter(x0,4,[c1 data_nm.c[2:end]])), data_nm.c)

# Zygote.hessian(((a, b),) -> a*b, [2, 3])

# using Zygote: @adjoint
# nestlevel() = 0
# @adjoint nestlevel() = nestlevel()+1, _ -> nothing
# function f(x)
#     println(nestlevel(), " levels of nesting")
#     return x^2
# end
# grad(f, x) = gradient(f, x)[1]
# grad(f,1)

		
		
# using Flux, Zygote, Optim, FluxOptTools, Statistics
# m      = Chain(Dense(1,3,tanh) , Dense(3,1))
# x      = LinRange(-pi,pi,100)'
# y      = sin.(x)
# loss() = mean(abs2, m(x) .- y)
# Zygote.refresh()
# pars   = Flux.params(m)
# lossfun, gradfun, fg!, p0 = optfuns(loss, pars)
# res = Optim.optimize(Optim.only_fg!(fg!), p0, Optim.Options(iterations=1000, store_trace=true))


# import Pkg; Pkg.add("DistributedArrays")

# using Distributed
# using DistributedArrays
# #t = @async addprocs(8)
# t = addprocs(8)
# nprocs()
# nworkers()

# d=dzeros(100,100)
# d[:L]

# rmprocs(workers())

# @everywhere using SharedArrays


# # Utilize workers as and when they come online
# if nprocs() > 1   # Ensure at least one new worker is available
   
# end

# if istaskdone(t)   # Check if `addprocs` has completed to ensure `fetch` doesn't block
#     if nworkers() == N
#         new_pids = fetch(t)
#     else
#         fetch(t)
#     end
# end

# S = SharedArray{Int,2}((3,4), init = S -> S[localindices(S)] = repeat([myid()], length(localindices(S))))
# 3×4 SharedArray{Int64,2}:

# length(Sys.cpu_info())
# Threads.nthreads()


# addprocs(8)

# #BLAS.set_num_threads(1)
# @everywhere using <modulename> or @everywhere include("file.jl").


# Sys.free_memory()/2^20

#  versioninfo(verbose=true)



			
			
			
			
			
			
			

## load data
xf = XLSX.readxlsx("node_stats_forsimulation_all.xlsx") #xf["BHCs"]["A1:I1"]
data = vcat( [(XLSX.eachtablerow(xf[s]) |> DataFrames.DataFrame) for s in XLSX.sheetnames(xf)]... )
unique!(data) # delete duplicate rows
#data[(isequal.(data.tkr,"JPM") .| isequal.(data.tkr,"AIG")) .& isequal.(data.qt_dt,192),:]


#= network primitives from the data
p_bar = total liabilities
c = outside assets
assets = total assets
w = net worth
b = outside liabilities
=#

data = data[isless.(0,data.w), :]
data = data[isequal.(data.qt_dt,192), :] # keep quarter == 195 = 2008q4
sort!(data, :assets, rev = true)
data = data[1:10,:] # keep small number of nodes, for testing
N = size(data,1) # number of nodes

# rescale units
units = 1e6;
data[:,[:w, :c, :assets, :p_bar, :b]] .= data[!,[:w, :c, :assets, :p_bar, :b]]./units

# fake missing data to make problem feasible
data.b[:] .= missing
data.c[:] .= missing

# create network variables
data.f = data.p_bar .- data.b # inside liabilities
data.d =  data.assets .- data.c # inside assets

# keep track of missing variables
col_with_miss = names(data)[[any(ismissing.(col)) for col = eachcol(data)]] # columns with at least one missing 
#data_nm = dropmissing(data, disallowmissing=true) # drop missing
data_nm = coalesce.(data, data.w .+ 0.01 .+ rand(N)) # replace missing by a value
nm_c = findall(x->x==0,ismissing.(data.c))
nm_b = findall(x->x==0,ismissing.(data.b))

# take a look
names(data) # column names
describe(data)

show(data, true)

# parameters
g0 = 0.0 # bankruptcy cost

function shock(c)
    # draw shocks that produce the probability of default δ 
    a = 1
    b = log.(data.delta)./(log.(1 .-data.w./c))
    dist = Beta.(a,[b...])
    draws = rand.(dist, 1)
    vcat(draws...).*c
    #rand(Float64,N) # normal
end

## Optimization

# initial values
D = 1 # number of draws
rng = MersenneTwister(1234);
dist = Beta.(2,fill(2,N,D))
x0 = rand.(rng,dist, 1)
x0 = vcat(x0...)
			
			
#x0 = fill(0.0,N,D)
A0 = rand(rng,N,N);[A0[i,i]=0.0 for i=1:N];A0=LowerTriangular(A0);

# set up optimization

#m = Model(Ipopt.Optimizer) # settings for the solver
m = Model(with_optimizer(Ipopt.Optimizer, start_with_resto="yes", linear_solver="mumps", max_iter=100000))
#m = optimizer_with_attributes(Ipopt.Optimizer, "start_with_resto" => "yes", "linear_solver"=>"mumps")

@variable(m, 0<=p[i=1:N,j=1:D]<=data.p_bar[i], start = data.p_bar[i]) 
@variable(m, 0<=c[i=1:N]<=data.assets[i], start = data_nm.c[i])  
@variable(m, 0<=b[i=1:N]<=data.p_bar[i], start = data_nm.b[i])   
@variable(m, 0<=A[i=1:N, j=1:N]<=1, start=A0[i,j])  # start=A0[i,j]

@constraint(m, sum(A,dims=2).*data.p_bar .== data_nm.f) # payments to other nodes add up to inside liabilities f
@constraint(m, A' * data.p_bar .== data_nm.d) # payments from other nodes add up to inside assets d

#fix(v::VariableRef, value::Number; force::Bool = false)
#delete(model, con)

# liabilities are net liabilities: A[i,i]=0 and A[i,j]A[j,i]=0
@constraint(m, [i = 1:N], A[i,i]==0)
for i=1:N
    j=1
    while j < i
        @complements(m, 0 <= A[i,j],  A[j,i] >= 0)
        j += 1
    end
end

# register max and min non-linear functions
maxfun(n1, n2) = max(n1, n2)
minfun(n1, n2) = min(n1, n2)

# chen-mangassarian smoothing function for max
# maxfun_cm(n1, n2,param=1000) = n1-n2 + log(1 + exp(-param*(n1-n2)))/param # converges to max when param \to \infty 
# minfun_cm(n1,n2,param=1/10000) = n1 - param* log(1+exp((n1-n2)/param)) # converges to min when param \to 0

# maxfun(n1,n2) = maxfun_cm(n1,n2,100) 
# minfun(n1,n2) = minfun_cm(n1,n2,1/1000)

# fisher-burmeister smoothing
# maxfun_fb(n1,n2,param=1/100000) = ( n1 + n2 + sqrt( (n1-n2)^2 + param^2 ) - param )/2  # converges to max as param \to 0

JuMP.register(m, :maxfun, 2, maxfun, autodiff=true)
JuMP.register(m, :minfun, 2, minfun, autodiff=true)

# clearing vector
myexpr = []
for j=1:D
    push!(myexpr, (1+g0)*(A'*p[:,j] .+ c .- x0[:,j].*c) .- g0.*data.p_bar)
end
    
@variable(m, aux[i=1:N,j=1:D])
for i = 1:N
    for j=1:D
        @constraint(m, aux[i,j] .== myexpr[j][i] )
    end
end

for i = 1:N
    for j=1:D
        @NLconstraint(m,  minfun(data.p_bar[i], maxfun(aux[i,j],0)) == p[i,j] ) 
    end
end

[fix(c[i], data.c[i]; force=true) for i  in nm_c]
[fix(b[i], data.b[i]; force=true) for i  in nm_b]

#@NLobjective(m, Max , sum(sum( x0[i,j]*c[i]+data.p_bar[i]-p[i,j] for i=1:N)/D for j=1:D) ) #*sum(x[i]+p_bar[i]-p[i] for i=1:N) 

@NLobjective(m, Max , sum(sum(-p[i,j] for i=1:N)/D for j=1:D) ) 

#unset_silent(m)
JuMP.optimize!(m)

termination_status(m)
objective_value(m)

psol = JuMP.value.(p)
csol = JuMP.value.(c)
bsol = JuMP.value.(b)
Asol = JuMP.value.(A)

			
Gadfly.spy(Asol)
Aplot = deepcopy(Asol)
Aplot[Aplot.<1e-3] .=0
			Aplot
