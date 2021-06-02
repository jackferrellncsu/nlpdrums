using Random
using StatsBase
using Distributions


function createData(n, p, seed)

    Random.seed!(seed)

    randMatrix = rand(Normal(0, 1), n, p)

    beta = rand([-4:4;], n, 1)

    raw = randMatrix * beta

    py = Sigmoid.(raw)

    y = py .> .5
    final = hcat(randMatrix, y)

    return beta, final
end

function Sigmoid(x)
    1 / (1 + exp(-x))
end
