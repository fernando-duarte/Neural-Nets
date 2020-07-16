using NLsolve
using Zygote
using Zygote: @adjoint, forward

using Pkg
Pkg.add("DistributionsAD")
using DistributionsAD

@adjoint nlsolve(f, j, x0; kwargs...) =
    let result = nlsolve(f, j, x0; kwargs...)
        result, function(vresult)
            # This backpropagator returns (- v' (df/dx)⁻¹ (df/dp))'
            v = vresult[].zero
            x = result.zero
            J = j(x)
            #_, back = forward(f -> f(x), f)
			_, back = pullback(f -> f(x), f)
            return (back(-(J' \ v))[1], nothing, nothing)
        end
    end



d, = gradient(p -> nlsolve(x -> [x[1]^3 - p], x -> fill(3x[1]^2, (1, 1)),[1.0]).zero[1],8.0)
d ≈ 1/3 * 8.0^(1/3 - 1)


using DataFrames, XLSX
using Missings
using LinearAlgebra, Random, Distributions, NLsolve, Complementarity
using Test
using NLopt
using Zygote

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

