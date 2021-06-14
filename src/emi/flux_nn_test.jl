using Flux
using LinearAlgebra
using Statistics

# Creating data based on train_size
train_size = 5000
real_data = generate_real_data(train_size)
fake_data = generate_fake_data(train_size)

# -------------------- Training The Model --------------------- #

# Organizing the data in batches, organizing data into dataset
X = hcat(real_data,fake_data)
Y = vcat(ones(train_size), zeros(train_size))
data = Flux.Data.DataLoader((X, Y'), batchsize=100, shuffle=true)

# Defining our model, optimization algorithm and loss function
# @function Descent - Classic gradient descent optimiser with learning rate η
nn = neural_net()
opt = Descent(0.05)
loss(x, y) = sum(Flux.Losses.binarycrossentropy(nn(x), y))

# Actual training
ps = Flux.params(nn)
epochs = 20
for i in 1:epochs
    Flux.train!(loss, ps, data, opt)
end
println(mean(nn(real_data)),mean(nn(fake_data)))
accuracy(x, y) = mean(onecold(nn(x), 1) .== onecold(y, 1))
accuracy(test, test[:, 1] .== field)
-----------------------------------------------------------------
# ------------------------- Functions ------------------------- #
-----------------------------------------------------------------

# generates "real" data to train using neural net
# @param n - length of matrix created
# @return matrix of dimensions 2 x n
function generate_real_data(n)
    x1 = rand(1, n) .- 0.5
    x2 = (x1 .* x1) * 3 .+ randn(1,n)*0.1
    return vcat(x1, x2)
end

# generates "fake" data to train using neural net
# @param n - length of matrix created
# @return matrix of dimensions 2 x n
function generate_fake_data(n)
    angle  = 2 * π * rand(1,n)
    r  = rand(1,n) / 3
    x1 = @. r * cos(angle)
    x2 = @. r * sin(angle) + 0.5
    return vcat(x1,x2)
end

# creation of neural network architecture
# @function Dense - takes in input, output, and activation
# function; creates dense layer based on parameters.
# @return nn - both dense layers tied together
function neural_net()
    nn = Chain(
            Dense(2, 25, relu),
            Dense(25, 1, x->σ.(x))
            )
    return nn
end

-----------------------------------------------------------------
# -------------------------- Visuals --------------------------- #
-----------------------------------------------------------------

# Original data
scatter(real_data[1,1:500],real_data[2,1:500])
scatter!(fake_data[1,1:500],fake_data[2,1:500])

# Model predictions
scatter(real_data[1,1:100],real_data[2,1:100], zcolor=nn(real_data)')
scatter!(fake_data[1,1:100],fake_data[2,1:100],zcolor=nn(fake_data)',legend=false)
