#required packages in file calling this one:
#Flux
#StatsBase
#Statistics
#Random

"""
    EmbeddingsTensor(data, context_size = 5, x_only = false)

Creates a 3-dimensional matrix of word embeddings
"""
function EmbeddingsTensor(data, word_embeddings_dict, context_size = 5, x_only = false)
    tensor = zeros(300, context_size, size(data)[1])

    result = zeros(300, size(data)[1])

    for (i, r) in enumerate(eachrow(data))
        sentence_mat = zeros(300, context_size)
        if length(r[1]) >= context_size && context_size != 1
            for (j, w) in enumerate(r[1][end-4:end])
                sentence_mat[:, j] = get(word_embeddings_dict, w, zeros(300))
            end
        else
            sent_length = length(r[1])
            for j in 1:context_size
                if j <= sent_length
                    sentence_mat[:, j] = get(word_embeddings_dict, r[1][j], zeros(300))
                else
                    sentence_mat[:, j] = mean.(eachrow(sentence_mat[:, 1:sent_length]))
                end
            end
        end
        tensor[:, :, i] = sentence_mat
        result[:, i] = get(word_embeddings_dict, r[2], zeros(300))
    end
    if !x_only
        return tensor, result
    else
        return tensor
    end
end
"""
   SampleMats(x_mat, y_mat, prop = 0.9)
 Splits matrices into specified groups, returns x_train, x_test, y_train, y_test
 in that order.  Assumes observations are columns and in same order in both inputs.
"""
function SampleMats(x_mat, y_mat, prop = 0.9)
    inds = [1:size(x_mat)[3];]
    length(inds)
    trains = sample(inds, Int(floor(length(inds) * prop)), replace = false)
    inds = Set(inds)
    trains = Set(trains)
    tests = setdiff(inds, trains)

    train_x = x_mat[:, :, collect(trains)]
    train_y = y_mat[:, collect(trains)]

    test_x = x_mat[:, :, collect(tests)]
    test_y = y_mat[:, collect(tests)]


    return train_x, test_x, train_y, test_y
end

"""
    TrainNN!(epochs, loss, nn, opt)

Trains a neural net using the specified number of epochs.  Uses input optimizer
and loss function.

Returns a trace plot.
"""
function TrainNN!(epochs, loss, nn, opt)
    traceY = []
    ps = Flux.params(nn)
    for i in 1:epochs
        Flux.train!(loss, ps, trainDL, opt)
        println(i)
        totalLoss = 0
        for (x,y) in trainDL
         totalLoss += loss(x,y)
         #println("Total Loss: ", totalLoss)
        end
        push!(traceY, totalLoss)
    end
    return traceY
end
