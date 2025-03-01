using Arpack
using Base.Threads
using BasisMatrices
using Dierckx
using Interpolations
using LinearAlgebra
using NLsolve
using Parameters
using ProgressMeter        # show progress of each loop with: @showprogress 1 "Computing..." for i in 1:50
using Plots
using QuantEcon
using Roots
using SparseArrays
using StatsBase


include("src/structs.jl")               # Structs for the parameters    
include("src/utils.jl")                 # Function to initialize the parameters
include("src/EGM.jl")                   # Function to compute the EGM policy functions
include("src/dist.jl")                  # Function to compute the transition matrix and stationary distribution
include("src/shock.jl")                 # Function to compute the shock processes

function Tax_solver!(pr::AiyagaryParam)
    @unpack Φ,EΦ,a_dist,z,nz,Ia,w,σ,γ,τ = pr
    luΦ = lu(Φ)
    diff_T = 1.0
    while diff_T > 1e-8
        solveλ_eg!(pr,luΦ,EΦ,pr.λcoef,1e-8)
        transitionMartixMaker!(pr)
        distFinder!(pr)

        c_pol = hcat([pr.cf[s](vec(a_dist)) for s in 1:nz]...)
        n_pol = ( (1-τ) .* w.*z'.* c_pol.^(-σ) ).^(1/γ)
    
        z_1 = kron(z,ones(Ia,1))
        T_new = dot(pr.dist,τ.*(w.*z_1.*reshape(n_pol,:,1)))
        diff_T = abs(T_new - pr.T)
        if !isnan(T_new)
            pr.T = T_new*0.9 + pr.T*0.1
        else
            error("T is NaN")  # Throws an error with the message
        end
    end
end

function βres(β)::Float64
    @unpack a_dist,nz,r,EΦ,τ,w,z,σ,γ,Ia = pr
    pr.β = min(β[1],1/(1+r))

    Tax_solver!(pr)

    c_pol = hcat([pr.cf[s](vec(a_dist)) for s in 1:nz]...)
    n_pol = ( (1-τ) .* w.*z'.* c_pol.^(-σ) ).^(1/γ)
    z_1 = kron(z,ones(Ia,1))

    return dot(pr.dist,kron(ones(pr.nz,1),a_dist))/dot(pr.dist,z_1.*reshape(n_pol,:,1))  - pr.K2N
end

function find_steady_statebyβ!(pr::AiyagaryParam)
    grid_maker!(pr)
    luΦ = lu(pr.Φ) 

    K2Y  = 10.8
    Y2K  = 1/K2Y
    pr.δ = pr.α * Y2K - pr.r
    pr.K2N = K2N = (Y2K/pr.Z)^(1/(pr.α-1)) 
    pr.w = (1-pr.α)*pr.Z*K2N^pr.α

    ret = nlsolve(βres,[pr.β])
    print("The error in K/N ratio is: ",βres(ret.zero))

    @unpack a_dist,nz,r,τ,w,z,σ,γ,Ia = pr
    c_pol = hcat([pr.cf[s](vec(a_dist)) for s in 1:nz]...)
    n_pol = ( (1-τ) .* w.*z'.* c_pol.^(-σ) ).^(1/γ)
    z_1 = kron(z,ones(Ia,1))
    pr.N = dot(pr.dist,z_1.*reshape(n_pol,:,1))
    pr.K = dot(pr.dist,kron(ones(pr.nz,1),a_dist))
end

function find_steady_statebyK!(pr::AiyagaryParam,shock_tax::Float64, tol=1e-5)
    @unpack EΦ,Φ,α,Z,δ,z,σ,γ,nz,β,Ia  = pr

    pr.τ = shock_tax
    grid_maker!(pr)
    luΦ = lu(Φ)
    diffK2N = Inf  # Initialize diffK to a large value
    iteration = 0  # Iteration counter
    
    min_k2n = ( (1/β-1+δ)/(α*Z) )^ (1/(α-1))

    while diffK2N > tol
        Tax_solver!(pr)
        c_pol = hcat([pr.cf[s](vec(a_dist)) for s in 1:nz]...)
        n_pol = ( (1-pr.τ) .* pr.w.*z'.* c_pol.^(-σ) ).^(1/γ)
        z_1 = kron(z,ones(Ia,1))

        K2N_supply = dot(pr.dist,kron(ones(pr.nz,1),a_dist))/dot(pr.dist,z_1.*reshape(n_pol,:,1))
        diffK2N = abs(K2N_supply - pr.K/pr.N)
        pr.N = dot(pr.dist,z_1.*reshape(n_pol,:,1))
        pr.K = max(pr.N*K2N_supply*0.4 + pr.K*0.6, min_k2n*pr.N)
        pr.K2N = K2N = pr.K / pr.N
        pr.w = (1 - α) * Z * K2N .^ α
        pr.r = α * Z * K2N .^ (α - 1) .- δ

        println("Iteration $iteration: Diff K = $diffK2N, K2N_sup=$(K2N_supply), pr.K2N_dem=$(pr.K/pr.N), T=$(pr.T)")
        iteration += 1
    end
end



function iterate_EGM_backward(pr::AiyagaryParam,λcoef::Vector{Float64},luΦ,EΦ::SparseMatrixCSC{Float64, Int64})
    @unpack na,nz,r,σ,a,z_p = pr
    IterateConsumption_λ!(pr,λcoef,EΦ)     
    λ = zeros((na+1)*nz) 
    # print("r=$(pr.r) , w=$(pr.w), tax=$(pr.τ), T=$(pr.T)\n")
    for s in 1:nz
        λ[(s-1)*(na+1)+1:s*(na+1)] = (1+r).*pr.cf[s](a).^(-σ) #compute consumption at gridpoints
    end
    
    return luΦ\λ,pr.cf
end

function iteratedistribution_forward!(pr,dist,dist_new,cf,r,w,T,τ)
    @unpack γ,σ,a_dist,a,nz,Ia,z_p = pr
    c_pol = hcat([cf[s](vec(a_dist)) for s in 1:nz]...)
    n_pol = ( (1-τ) .* w.*z'.* c_pol.^(-σ) ).^(1/γ)
    a_pol = (1+r).*a_dist .+ w.*z'.*n_pol.*(1-τ)  - c_pol .+ T
    a_pol = max.(min.(a_pol,a_dist[end]),a_dist[1]);

    temp = [kron(z_p[s,:],BasisMatrix(Basis(SplineParams(a_dist,0,1)),Direct(),@view a_pol[:,s]).vals[1]') for s in 1:nz]
    Λ = hcat(temp...)

    mul!(dist_new,Λ,dist)
    dist_new = dist_new ./ sum(dist_new)
end

function compute_policy_path(pr_initial, pr_final,shock,Rt,Wt,Tt,τt)
    @unpack EΦ,Φ,γ,σ,w,r,a_dist,na,nz,Ia, = pr_final
    @unpack T = shock
    luΦ = lu(Φ)
    
    for t in reverse(6:T-5)
        pr.r = Rt[t]
        pr.w = Wt[t]
        pr.τ = τt[t]
        pr.T = Tt[t]
        λcoefs[:,t],cfs[t] = iterate_EGM_backward(pr,λcoefs[:,t+1],luΦ,EΦ)  
    end

    return λcoefs,cfs
end


function valuefunction_calculator(pr_initial,pr)
    @unpack nz,a_dist = pr
    tol = 1e-15
    c = hcat([pr_initial.cf[s].(vec(a_dist)) for s in 1:nz]...)
    V = u.(c,[pr])
    diff_v = 1.
    while diff_v > tol
        V_new = u.(c,[pr]) .+ β.*V
        diff_v = norm(V_new .- V,Inf)
        V .= V_new
    end
    return V
end


