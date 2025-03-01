function grid_maker!(pr)
    pivot = 0.25
    a = exp10.(range(log10(pivot), log10(pr.max_a - pr.min_a + pivot), length=pr.na) ) .+ pr.min_a .- pivot
    a[1]   = pr.min_a
    a[end] = pr.max_a
    a_dist = exp10.(range(log10(pivot), log10(pr.max_a - pr.min_a + pivot), length=pr.Ia) ) .+ pr.min_a .- pivot
    a_dist[1]   = pr.min_a
    a_dist[end] = pr.max_a
    mc = rouwenhorst(pr.nz, pr.ρe, pr.σe)

    z = exp.(mc.state_values)
    pr.z_p = z_p = mc.p
    π_stationary = real(eigs(z_p',nev=1)[2])
    π_stationary ./= sum(π_stationary)
    pr.z = z./dot(π_stationary,z)

    abasis = Basis(SplineParams(a,0,2))
    a = nodes(abasis)[1]'

    xvec = LinRange(0,1,pr.Ia).^2.  #The Na -1 to to adjust for the quadratic splines
    pr.a_dist = a_dist = pr.min_a .+ (pr.max_a - pr.min_a).*xvec; #nonlinear grid knots
    pr.dist = ones(pr.Ia*pr.nz)/(pr.Ia*pr.nz);

    xvec = LinRange(0,1,pr.na).^2.5  #The Na -1 to to adjust for the quadratic splines
    a = pr.min_a .+ (pr.max_a - pr.min_a).*xvec #nonlinear grid knots
    pr.abasis = abasis = Basis(SplineParams(a,0,2))
    pr.a = nodes(abasis)[1];

    pr.Φ = kron(Matrix(I,pr.nz,pr.nz),BasisMatrix(pr.abasis,Direct()).vals[1])
    pr.EΦ = kron(pr.z_p,BasisMatrix(pr.abasis,Direct(),pr.a).vals[1])
    pr.λcoef = pr.Φ\repeat((1/pr.β).*((1-pr.β)*pr.a .+1).^(-pr.σ),pr.nz);
end

function u(c,pr)
    @unpack σ = pr
    return c^(1-σ) / (1-σ)
end

function uinv(v,pr)
    @unpack σ = pr
    return ( v*(1-σ) )^(1/(1-σ))
end
