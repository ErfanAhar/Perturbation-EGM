function transitionMartixMaker!(pr::AiyagaryParam)::SparseMatrixCSC{Float64, Int64}
    @unpack γ,σ,w,r,τ,T,a_dist,a,nz,Ia,z_p,EΦ,λcoef,cf = pr
    
    c_pol = hcat([cf[s](vec(a_dist)) for s in 1:nz]...)
    n_pol = ( (1-τ) .* w.*z'.* c_pol.^(-σ) ).^(1/γ)
    a_pol = (1+r).*a_dist .+ w.*z'.*n_pol.*(1-τ) - c_pol .+ T
    a_pol = max.(min.(a_pol,a_dist[end]),a_dist[1]);

    temp = [kron(z_p[s,:],BasisMatrix(Basis(SplineParams(a_dist,0,1)),Direct(),@view a_pol[:,s]).vals[1]') for s in 1:nz]
    pr.Λ = hcat(temp...)
end

function distFinder!(pr::AiyagaryParam)
    @unpack nz,Ia,Λ,dist = pr
    diff = 1
    tol = 1e-15
    dist_2 = copy(dist)
    i = 1
    while diff > tol
        if i%2 == 0
            mul!(dist_2,Λ,dist)
        else
            mul!(dist,Λ,dist_2)
        end
        if i%100 == 0
            diff = norm(dist_2 - dist, Inf)
        end
        i+=1
    end
    pr.dist = dist ./sum(dist)
end