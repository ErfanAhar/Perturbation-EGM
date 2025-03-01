function shock_perm_maker!(pr)
    @unpack T,value = pr
    shock = ones(T)
    shock[6:end] .= 1+value 
    pr.shock = shock
end

function shock_temp_maker!(pr)
    @unpack T,value,ρ = pr
    shock = ones(T)
    shock[6] = shock[5] + value*shock[5]
    for t in 7:pr.t
        shock[t] = 1 - ρ*(1-shock[t-1])
    end
    pr.shock = shock
end