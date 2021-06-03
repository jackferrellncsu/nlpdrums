using LinearAlgebra
using Plots
using Random
using StatsBase
using Distributions

n = 1000
beta = [1,2,3,4,5]
p = length(beta)
X = rand(Normal(0, 1), n, p)
py = 1 ./ ( 1 .+ exp.(-X * beta))
y = py .> rand(n)

counter = 0
loss = zeros(200000)
w = rand(p) .* 2*sqrt(6/p) .+ (-sqrt(6/p))
aaa = 1
lr = .01
while abs(aaa) > 10^(-30)

    local a = (1 ./ (1 .+exp.(-X*w))) .- y
    local nabla = sum(Diagonal(vec(a)) * X, dims = 1)
    global w = w .- lr .* nabla'
    global counter +=1
    local sig = (1 ./ (1 .+exp.(-X*w)))
    global loss[counter] = sum(-(y .* log.(sig) + (1 .- y) .* log.(1 .- sig)))

    if counter > 1
        global aaa = loss[counter-1] - loss[counter]
        if aaa < 0
            global lr = lr * .9
        end
    end
end
