using Flux
using CUDA
using Random
using Flux.Losses
using Flux.Data


loss(x, y) = binarycrossentropy(nn(x), y)

train_x = rand(Float32, 10, 20)
train_y = .5 .>= rand(20)

test_x = rand(Float32, 10, 20)
test_y = .5 .>= rand(20)

trainDL = DataLoader((train_x, train_y'))
testDL = DataLoader((test_x, test_y'))

opt = RADAM()

nn = Chain(
    Dense(10, 1, x->σ.(x))
) |> gpu

Flux.train!(loss, params(nn), trainDL, opt)
