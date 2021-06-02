using TextAnalysis
include("DTMCreation.jl")
include("syntheticData.jl")


data = importClean()

dtm = CreateDTM(data, " Bariatrics")
dtm = transpose(dtm)

w = zeros(size(dtm)[2])
x = dtm[1,1:end - 1]

lr = 0.
test1 = StochGradientDescent(dtm, lr)

b, mat = createData(3, 3, 6)


test2 = StochGradientDescent(mat, 0.01)
#Inputs:
#dtm = document term matrix
#lr = learning rate
function StochGradientDescent(dtm, lr)
    #Weights are all elements of vector instead of last
    #Initialize weights at 0
    w = zeros(size(dtm)[2])
    loss = Vector{Tuple{Float64, Int64}}()
    #Initialize
    epsilon = 10000

    #error testing
    count = 0
    while (epsilon > 0.0001) #or when loss starts going up on held out set
        shuffle!(dtm)
        for doc in eachrow(dtm)
            #Initialize x
            x = doc[1:end]

            #calculate
            #= | |\
              || |_
            =#
            push!(loss, (CalculateLoss(x, w), count))

            # if (count >= 1) && (loss[end] > loss[end - 1])
            #     lr *= .75
            # end
            #calculate gradient
            gradient = CalculateGradient(x, w)

            #Update w based on learning rate and gradient
            w = w .- (lr .* gradient)

            #check convergence condition
            epsilon = sqrt(sum(gradient.^2))

            if count % 10 == 0
                println()
                print(epsilon, loss[end], lr)
            end

            if epsilon < 0.0001
                return w
            end


            count+=1

        end

        end
    end
    return w
end

#inputs are
#w = vector of weights
#x = vector of inputs
#y = true outcome
#This doesnt work
function CalculateLoss(w, x)
    y = x[end]
    b = w[end]
    sig = Sigmoid(sum(w[1:end - 1] .* x[1:end-1]) + b)
    return -(y * log(sig) + (1-y) * log(1 - sig))
end

function Sigmoid(x)
    1 / (1 + exp(-x))
end

#calculates the gradient
#x is a row from the DTM (still includes label)
#w is a vector of weights
function CalculateGradient(x, w)
    grads = []
    y = x[end]
    b = w[end]
    sig = Sigmoid(sum(w[1:end - 1] .* x[1:end-1]) + b)-y
    for i in x[1:end - 1]
        push!(grads, sig*i)
    end

    return push!(grads, sig)
end

#= | ||
     |_
=#
