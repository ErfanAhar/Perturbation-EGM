
function IterateConsumption_λ!(pr::AiyagaryParam,λcoefs::Vector{Float64},EΦ::SparseMatrixCSC{Float64, Int64})
    @unpack σ,β,γ,w,r,τ,nz,na,min_a,a,z,T,min_c = pr
    Eλ = reshape(EΦ*λcoefs,:,nz) #precomputing expectations
    cEE = (β.*Eλ).^(-1/σ) #consumption today
    nEE = ((1-τ).*w.*z' .* (cEE).^(-σ) ).^(1/γ) #labor supply today
    Implieda = (a .+ cEE .- T .- (w.*z'.*nEE).*(1-τ))/(1+r)  #Implied assets today
    
    cf = Vector{Spline1D}(undef,nz)#implied policy rules for each productivity
    a_cutoff::Dict{Float64,Float64} = Dict{Float64,Float64}() #Stores the points at which the borrowing constraint binds

    for s in 1:nz
        #with some productivities the borrowing constraint does not bind
        if issorted(Implieda[:,s])
            p = 1:na+1
        else
            p = sortperm(Implieda[:,s])
        end
        if Implieda[p[1],s] > min_a #borrowing constraint binds
            a_cutoff[z[s]] = Implieda[p[1],s]
            #add extra points on the borrowing constraint for interpolation
            â = [min_a;Implieda[p,s]]
            # ĉ = [ ((1-τ)*w*z[s])^((1+γ)/(γ+σ));cEE[p,s]]
            ĉ = [ min_c[s];cEE[p,s]]
            cf[s] = Spline1D(â,ĉ,k=1)
        else
            a_cutoff[z[s]] = -Inf
            cf[s] = Spline1D(Implieda[p,s],cEE[p,s],k=1)
        end
    end
    pr.cf = cf
end


function elastic_iterateλ_eg!(pr::AiyagaryParam,Φ,EΦ::SparseMatrixCSC{Float64, Int64},λcoefs::Vector{Float64})
    @unpack na,nz,r,σ,a = pr
    IterateConsumption_λ!(pr,λcoefs,EΦ)
    λ = zeros((na+1)*nz) 
    
    for s in 1:nz
        λ[(s-1)*(na+1)+1:s*(na+1)] = (1+r).*pr.cf[s](a).^(-σ) #compute consumption at gridpoints
    end
    λcoefs_new = (Φ\λ);
    diff = norm(λcoefs .- λcoefs_new,Inf)
    λcoefs .= λcoefs_new
    return diff
end

function solveλ_eg!(pr::AiyagaryParam,Φ,EΦ,λcoefs,tol=1e-8)
    #compute the minimum consumption in constraint
    minimum_Consumption!(pr)
    #increases stability to iterate on the time dimension a few times
    diff = 1.
    while diff > tol           
        #then use newtons method
        diff = elastic_iterateλ_eg!(pr,Φ,EΦ,λcoefs)
    end
    pr.λcoef = λcoefs
end

function minimum_Consumption!(pr::AiyagaryParam)
    @unpack nz,z,w,σ,γ,τ,T  = pr
    f(c) = c .- (1-τ).*w.*z.* ((1-τ).*w.*z.*c.^(-σ)).^(1/γ) .- T
    # Solve the system of equations
    sol = nlsolve(f, ones(7))
    pr.min_c = sol.zero
end