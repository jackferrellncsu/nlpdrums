using Plots
# define the function
function f(x)
    return x^4 - 2x^2 - x
end

# define the gradient
function gradient(x)
    return 4x^3 - 4x - 1
end

function approxGradient(x,h)
    return (f(x+h)-f(x-h))/(2*h)
end

function plotGif(xaxis,yaxis,startx)
    Plots.plot(xaxis,yaxis,leg = false)
    Plots.plot!([startx],[f(startx)], seriestype = :scatter, leg = false)
end


xaxis = []
yaxis = []

for i in -400:400
    push!(xaxis, i/200)
    push!(yaxis,f(i/200))
end

steps = []
loss = []

startx = -1.5
srate = .0095
lamb = 1
gamma = .9 # DONT CHANGE
v = 0
m = 0
b1 = .9 # DONT CHANGE
b2 = .999 # DONT CHANGE
e = 10^-8

anim = @animate for i ∈ 1:100
    println(i)
    plotGif(xaxis,yaxis,startx)


#Some Gradient Optimixations
    #= Leon Bottou Formula
    rate = srate*(1+srate*lamb*i)^(-1)
    global startx = startx - rate*gradient(startx)=#


    #= #Momentum Descent
    global v = gamma*v+srate*gradient(startx)
    global startx = startx - v =#

    #= Nesterov Accelerated Gradient
    global v = gamma*v+srate*gradient(startx-v)
    global startx = startx - v=#

    #= ADAM
    global m = b1*m + (1-b1)gradient(startx)
    global v = b2*v + (1-b2)gradient(startx)^2
    mhat = m / (1-b1^i)
    vhat = v/ (1-b2^i)
    global startx = startx - srate*mhat/(sqrt(vhat)+e)=#

    #=NADAM WIP
    global m = b1*m + (1-b1)*gradient(startx)
    mhat = m/(1-b1^i)
    global v = gamma*v+srate*gradient(startx-v)
    vhat = v/ (1-b2^i) #Not sure about this line
    global startx = startx - srate*(b1*mhat+(gradient(startx)*(1-b1))/(1-b1^i))/(sqrt(vhat)+e)
    =#

    global v = gamma*v+srate*gradient(startx-v)
    global startx = startx - v


    push!(loss,f(startx))
    push!(steps,i)

end

gif(anim, "anim_fps15.gif", fps = 100)
