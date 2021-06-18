using Flux

# @function - creates a vector (whole data set) of vectors (each individual sample)
# of vectors (the individual values inside the samples).
# @param num_samples - number of data samples you want. More samples, higher accuracy
# @return - two vectors inside of a vector. [data] represents the data sample(s) and
# [labels] represents the sum of each data sample
function generate_rnn_data(num_samples)
    data = []
    labels = []
    for i in 1:num_samples
        subdata = []
        sum = 0
        for ii in 1:rand(2:7)
            r = 10*rand(Float32)
            push!(subdata, [r])
            sum += r
        end
    push!(data,subdata)
    push!(labels, sum)
    end
    return [data,labels]
end

# Represents the loss of each individual value
function lossSig(x,y)
    ypred = rnn.(x)[end][1]
    l = abs(ypred - y)
    Flux.reset!(rnn)
    return l
end

# Total loss on function
function loss(X,Y)
    sum = 0
    for i in 1:length(Y)
        sum += lossSig(X[i],Y[i])
    end
    return sum
end

data, labels = generate_rnn_data(2000)

rnn = Flux.RNN(1,1,(x->x))

ps = Flux.params(rnn)

opt = Flux.ADAM()

epochs = 100
for i in 1:epochs
    println(i, " : ", loss(data,labels))
    Flux.train!(lossSig , ps , zip(data,labels) , opt)
end

Flux.reset!(rnn)
print(rnn(Float32[1]) + rnn(Float32[5]) + rnn(Float32[7]))
