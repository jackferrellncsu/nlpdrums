using TextAnalysis
using Plots
include("DTMCreation.jl")
include("syntheticData.jl")


data = importClean()

dtm = CreateDTM(data, " Bariatrics")
dtm = transpose(dtm)

w = zeros(size(dtm)[2])
x = dtm[1,1:end - 1]

lr = 0.1
test1 = StochGradientDescent(dtm, lr)

b, mat = createData(10000, 10, 420)


test2, losses, steps = StochGradientDescent(mat, .1, 1*10^-10)
#Inputs:
#dtm = document term matrix
#lr = learning rate
function StochGradientDescent(dtm, lr, ϵ)
    #Weights are all elements of vector instead of last
    #Initialize weights at 0
    w = zeros(size(dtm)[2])
    steps, losses, count = [], [], 0
    while true #Broken by return statement for epsilon

        #choosing a data point at random
        x = dtm[rand(1:size(dtm)[1];),1:end]

            #calculate
            #= | |\
              || |_
            =#
        push!(losses, CalculateLoss(x, w))
        push!(steps,count)


        #Adjusting Learning Rate
        if (count >= 1) && (losses[end] > losses[end - 1])
                lr *= .999
        end

        #calculate gradient
        gradient = CalculateGradient(x, w)

        #Update w based on learning rate and gradient
        #For Loop Versiom
        #=for i in 1:length(w)
            w[i] = w[i] - lr*gradient[i]
        end=#

        #Broadcasting Version
        w = w .- (lr .* gradient)

        #Calculate epsilon
        epsilon = sqrt(sum(gradient.^2))


        #print(epsilon, loss[end], lr)
        #Increment steps
        count+=1

        #Print Results
        if count%10 ==0
            println("Step: " * string(count) * ", Loss: " * string(losses[end]) * ", ϵ: " * string(epsilon) * ", η: "*string(lr))
        end

        #Check convergence conditions
        if epsilon < ϵ
            println("Step: " * string(count) * ", Loss: " * string(losses[end]) * ", ϵ: " * string(epsilon) * ", η: "*string(lr))
            return w, losses, steps
        end

        end

    end


#inputs are
#w = vector of weights
#x = vector of inputs
#y = true outcome
#This doesnt work
function CalculateLoss(x, w)
    y = x[end]
    b = w[end]
    #=
    sig = 0
    for i in 1:length(w[1:end-1])
        sig += w[i] * x[i]
    end
    sig += b
    sig = Sigmoid(sig)
    =#
    sig = Sigmoid(sum(w[1:end - 1] .* x[1:end - 1]) + b)
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
    sig = Sigmoid(sum(w[1:end - 1] .* x[1:end - 1]) + b)-y
    for i in x[1:end - 1]
        push!(grads, sig*i)
    end

    return push!(grads, sig)
end


#Making the Trace Plots

rolledlosses = []
for i in 50:(length(losses)-50)
    push!(rolledlosses, sum(losses[(i-49):(i+50)])/100)
end
Plots.plot(steps,losses, label = "Stochiastic Loss")
Plots.plot!(steps[1:length(rolledlosses)], rolledlosses, label = "Rolling Average over 50")
xlabel!("Steps")
ylabel!("Loss")
