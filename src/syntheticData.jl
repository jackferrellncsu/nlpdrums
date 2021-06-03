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

function Sigmoid(x)
    1 / (1 + exp(-x))
end
