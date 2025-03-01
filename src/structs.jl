@with_kw mutable struct AiyagaryParam
    # Households
    β::Float64  = 0.9751578966450003   # Discount factor 
    σ::Float64  = 5         # Inverse IES
    γ::Float64  = 10        # Inverse labor elasticity
    τ::Float64  = 0.1
    min_a::Float64 = 0     # borrowing limit for asset
    max_a::Float64 = 500  # max limit for illiquid asset
    # Labor Unions
    ϕ::Float64  = 2.073    # Disutility of labor
    ν::Float64  = 1        # Inverse Frisch elasticity
    # Firms
    Z::Float64  = 1        # Agg TFP
    α::Float64  = 0.36     # Capital share
    δ::Float64  = 0.0232   # Depreciation
    # Prices
    N::Float64  = 1        # Agg Labor
    K::Float64  = 41.18363 # Agg Capital
    r::Float64  = 0.01     # Real interest rate
    w::Float64  = 2.44        # Real Wages
    T::Float64  = 0.1792
    K2N::Float64 = 41.18362947600099
    # Grids 
    na::Int = 100 -1       # Number of grid points for asset
    Ia::Int = 1000          # number of grid points for agent distribution 
    nz::Int = 7            # Number of grid points for idiosyncratic income process 
    ρe::Float64 = 0.966    # Autocorrelation of earnings
    σe::Float64 = 0.13     # Cross-sectional std of log earnings
    a::Vector{Float64} = zeros(1)
    z::Vector{Float64} = zeros(1)
    a_dist::Vector{Float64} = zeros(1)
    z_p::Array{Float64, 2}  = zeros(1,1)
    # basis
    abasis::Basis{1, Tuple{SplineParams{Vector{Float64}}}} = Basis(SplineParams(collect(LinRange(0,1,60)),0,2))
    Φ::SparseMatrixCSC{Float64,Int64} = spzeros(na,na)
    EΦ::SparseMatrixCSC{Float64,Int64} = spzeros(na,na) # to compute expectations of marginal utility at gridpoints on potential savings
    # luΦ = lu(Φ)
    # stationary dist
    Λ::SparseMatrixCSC{Float64,Int64} = spzeros(Ia,Ia)
    dist::Vector{Float64}  = ones(1)/1000
    λcoef::Vector{Float64} = zeros(1)
    cf::Vector{Spline1D} = Vector{Spline1D}(undef,1)
    # minimum consumption
    min_c::Vector{Float64} = zeros(1)
end

@with_kw mutable struct ShockParam
    # Shock
    value::Float64 = -0.1
    T::Int64 = 300
    ρ::Float64 = 0.966    # Autocorrelation of shock
    shock::Vector{Float64}    = zeros(1)
    path_K2N::Vector{Float64} = zeros(1)
    path_T::Vector{Float64}   = zeros(1)
    λcoefs::Array{Float64, 2} = zeros(1,1)
    cfs::Array{Float64, 2}    = zeros(1,1)
end


