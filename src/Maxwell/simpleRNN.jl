using Flux

function generate_data(num_samples)
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

function lossSig(x,y)
    ypred = rnn.(x)[end][1]
    l = abs(ypred - y)
    Flux.reset!(rnn)
    return l
end

function loss(X,Y)
    sum = 0
    for i in 1:length(Y)
        sum += lossSig(X[i],Y[i])
    end
    return sum
end

data, labels = generate_data(1000)

rnn = Flux.RNN(1,1,(x->x))

ps = Flux.params(rnn)

opt = Flux.ADAM()

<<<<<<< Updated upstream
epochs = 1000
for i in 1:epochs
    println(i, " : ", loss(data,labels))
=======
epochs = 100
err = []
for i in 1:epochs
    e = loss(data,labels)
    push!(err,e)
    println(i, " : ", e)
>>>>>>> Stashed changes
    Flux.train!(lossSig , ps , zip(data,labels) , opt)
end

Flux.reset!(rnn)
rnn(Float32[-2])
rnn(Float32[3])
rnn(Float32[-5])
ps
