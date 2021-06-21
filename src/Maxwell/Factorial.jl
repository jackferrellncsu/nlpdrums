function fact(x)
    if x == 0
        return 1
    else
        return x*fact(x-1)
    end
end

println(fact(20))
