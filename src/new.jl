using LinearAlgebra
using Plots
import("syntheticData.jl")


X = rand(Normal(0, 1), 100, 10)

beta = rand([-8:8;], 10, 1)
#beta = ones(p,1)
raw = X * beta

py = []

for i in 1:length(raw)
    push!(py,Sigmoid(raw[i]))
end

y = py .> .5


GradDesc(X, y, .1)
#X = matrix of observed predictors
#y = vector of observed outcomes
function GradDesc(X, y, lr)
    #Initialize weights
    w = zeros(size(X[2]))

    #replace w/ sigmoids
    a = (1 ./ (1 .+exp.(-X*w))) .- y

    nabla = sum(Diagonal(a) * X, 1)

    w = w .- lr .* nabla

    return w
end

using Random
using StatsBase
using Distributions


function createData(n, p, seed)

    Random.seed!(seed)

    randMatrix = rand(Normal(0, 1), n, p)

    beta = rand([-8:8;], p, 1)
    #beta = ones(p,1)
    raw = randMatrix * beta

    py = []

    for i in 1:length(raw)
        push!(py,Sigmoid(raw[i]))
    end

    y = py .> .5
    final = hcat(randMatrix, y)

    return beta, final
end

counter = 0
loss = zeros(2000)
w = zeros(size(X)[2])
aaa = 1
while aaa > 10^(-6)

#replace w/ sigmoids
    a = (1 ./ (1 .+exp.(-X*w[1:10]))) .- y

    nabla = sum(Diagonal(a) * X, dims = 1)

    global w[1:10] = w[1:10] .- lr .* nabla[1:10]
    global counter +=1

    sig = (1 ./ (1 .+exp.(-X*w[1:10])))

    global loss[counter] = sum(-(y .* log.(sig) + (1 .- y) .* log.(1 .- sig)))

    if counter > 1
        global aaa = loss[counter-1] - loss[counter]
    end

end
