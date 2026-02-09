using SparseArrays
using LinearAlgebra
using Arpack
using Statistics
using Random
using DelimitedFiles
using NPZ
using ExpmV
using Dates
using ArgParse

Random.seed!(Dates.now().instant.periods.value)

function random_product_state(L::Int)
    ψ = nothing
    for i in 1:L
        θ1, θ2 = rand() * π, rand() * π
        ϕ1, ϕ2 = rand() * 2π, rand() * 2π
        c1 = cos(θ1 / 2)
        c2 = exp(im * ϕ1) * sin(θ1 / 2) * sin(θ2 / 2)
        c3 = exp(im * ϕ2) * sin(θ1 / 2) * cos(θ2 / 2)
        site_state = [c1, c2, c3]
        ψ = i == 1 ? site_state : kron(ψ, site_state)
    end
    return ψ / norm(ψ)
end

function superposition_state(L::Int)
    # single-site state (|0> + |1> + |2>) / sqrt(3)
    φ = ones(ComplexF64, 3) / sqrt(3)

    ψ = φ
    for _ in 2:L
        ψ = kron(ψ, φ)
    end

    return ψ
end

# ---------------------------------------------------------
# Haar-random unitary (3×3), converted to sparse
# ---------------------------------------------------------
function haar_unitary_sparse(n::Int)
    A = randn(ComplexF64, n, n)
    Q, R = qr(A)
    phases = Diagonal(R) ./ abs.(Diagonal(R))
    return sparse(Q * Diagonal(phases))
end

# ---------------------------------------------------------
# Build 9×9 Z3-conserving Haar unitary as sparse blockdiag
#
# Basis ordering (Z3 sectors):
#   charge 0: 00, 12, 21
#   charge 1: 01, 10, 22
#   charge 2: 02, 20, 11
#
# So U_z3 is blockdiag(U0, U1, U2)
# ---------------------------------------------------------
function z3_haar()
    U0 = haar_unitary_sparse(3)
    U1 = haar_unitary_sparse(3)
    U2 = haar_unitary_sparse(3)
    return blockdiag(U0, U1, U2)  # built-in sparse blockdiag
end

# ---------------------------------------------------------
# Sparse permutation matrix: Z3 basis → standard lexicographic basis
#
# Z3 basis order:
#   ["00","12","21",  "01","10","22",  "02","20","11"]
#
# Standard basis order:
#   ["00","01","02",  "10","11","12",  "20","21","22"]
# ---------------------------------------------------------
function permutation_matrix_sparse()
    z3  = ["00","12","21","01","10","22","02","20","11"]
    std = ["00","01","02","10","11","12","20","21","22"]

    rows = Int[]
    cols = Int[]
    vals = ComplexF64[]

    for (i, state) in enumerate(z3)
        j = findfirst(==(state), std)
        push!(rows, j)
        push!(cols, i)
        push!(vals, 1.0 + 0im)
    end

    return sparse(rows, cols, vals, 9, 9)
end

# ---------------------------------------------------------
# Apply sparse basis change: U_std = P * U_z3 * P'
# ---------------------------------------------------------
function transform(U_z3)
    P = permutation_matrix_sparse()
    return P * U_z3 * P'
end

function odd_layer(L::Int)
    U = transform(z3_haar())
    for _ in 3:2:L-1
        U = kron(U, transform(z3_haar()))
    end
    return U
end

function even_layer(L::Int)
    id = sparse(ComplexF64[1 0 0; 0 1 0; 0 0 1])
    U = id
    for _ in 2:2:L-1
        U = kron(U, transform(z3_haar()))
    end
    return kron(U, id)
end

function time_evolution(ψ::Vector{ComplexF64}, L)

    U_odd  = odd_layer(L)
    U_even = even_layer(L)

    ψ_odd = U_odd * ψ
    ψ_even = U_even * ψ_odd

    return normalize!(ψ_even)
end

function entropy_vn(ψ::Vector{<:Complex}, L::Int, subsystem::AbstractArray{Int})
    cut = length(subsystem)
    dimA = 3^cut
    dimB = 3^(L - cut)
    
    ψ_matrix = reshape(ψ, (dimA, dimB))
    svals = svdvals(ψ_matrix)

    
    S = 0.0
    
    for s in svals
        if s > 1e-15
            p = abs2(s)
            S -= p * log(p)
        end
    end
    return S
end

function get_z3_operators()
    id = sparse(ComplexF64[1 0 0; 0 1 0; 0 0 1])
    τ = sparse(ComplexF64[1 0 0; 0 exp(2im * pi / 3) 0; 0 0 exp(4im * pi / 3)]) ## τ in Romain's paper

    return id, τ
end

function build_term(operators::Vector{<:SparseMatrixCSC}) #
    term = operators[1]
    for j in 2:length(operators)
        term = kron(term, operators[j])
    end
    return term
end

function create_local_q_operator(L::Int, site::Int)
    id, τ = get_z3_operators()
    ops = fill(id, L)
    ops[site] = (im / sqrt(3)) * (τ' - τ)
    return build_term(ops)
end

function charge_expectation(ψ::Vector{ComplexF64}, L::Int)
    ω = cis(2pi / 3)   
    expU = zero(ComplexF64)

    for (idx, amp) in enumerate(ψ)
        n = idx - 1
        q = 0
        @inbounds for _ in 1:L
            q += n % 3
            n ÷= 3
        end
        expU += abs2(amp) * ω^q
    end

    return expU
end

function shannon_entropy(ψ::Vector{ComplexF64}, L::Int)
    # Probabilities for the three Z₃ charge sectors
    p0 = 0.0
    p1 = 0.0
    p2 = 0.0

    for (idx, amp) in enumerate(ψ)
        # Decode basis index -> total charge mod 3
        n = idx - 1
        q = 0
        for _ in 1:L
            q += n % 3
            n ÷= 3
        end
        q %= 3

        # Born probability
        w = abs2(amp)

        if q == 0
            p0 += w
        elseif q == 1
            p1 += w
        else
            p2 += w
        end
    end

    # Shannon entropy
    S = 0.0
    if p0 > 1e-15; S -= p0 * log(p0); end
    if p1 > 1e-15; S -= p1 * log(p1); end
    if p2 > 1e-15; S -= p2 * log(p2); end

    return S
end


function Entropy_t(L::Int, T::Float64, dt::Float64, p::Float64, shot::Int)
    
    # Initialize state
    #s_t = random_product_state(L)
    s_t = superposition_state(L)
    
    # Time evolution

    ω = exp(2im * pi / 3)
    
    # Build a list of single-site Q operators for measurement
    Ql = [create_local_q_operator(L, i) for i in 1:L]
    
    # Initialize lists to store results
    #S_list = Float64[]
    SE_list = Float64[]

    steps = Int(floor(T / dt))

    for _ in 1:steps
        
        # Record half-chain entropy
        #push!(S_list, entropy_vn(s_t, L, 1:L÷2))
        push!(SE_list, shannon_entropy(s_t, L))

        # Time evolution
        s_t = time_evolution(s_t, L)

        # Measurements
        if p != 0
            for l in 2:L-1 ## avoid edges
                if rand() < p ## edit this
                    p_m_zero  = real(s_t' * s_t) - real(s_t' * Ql[l] * Ql[l] * s_t)
                    p_m_one = 0.5 * real(s_t' * Ql[l] * s_t) + 0.5 * real(s_t' * Ql[l] * Ql[l] * s_t)
                    x1 = rand()
                    if x1 < p_m_zero
                        s_t = (s_t - (Ql[l] * Ql[l] * s_t)) / sqrt(p_m_zero)
                    elseif p_m_zero ≤ x1 < (p_m_one + p_m_zero)
                        s_t = 0.5 * (Ql[l] * s_t + Ql[l] * Ql[l] * s_t) / sqrt(p_m_one)
                    else
                        s_t = 0.5 * ((Ql[l] * Ql[l] * s_t) - s_t) / sqrt(1 - p_m_zero - p_m_one)
                    end
                end
            end
        end
    end

    """
    # Data storage for Jed's Mac
    base_folder = "/Users/uditvarma/Project_Data/z3-haar" 

    today_date = Dates.format(Dates.today(), "yyyy-mm-dd")
    parent_dir = dirname(base_folder)
    folder_name = basename(base_folder) * "_" * today_date
    folder = joinpath(parent_dir, folder_name)
    mkpath(folder)    
    filename_entropy = joinpath(folder, "L$(L),T$(T),dt$(dt),p$(p),dirQ,s$(shot)_hc.npy")
    npzwrite(filename_entropy, S_list)
    """

    """
    # Data storage for Udit's Mac
    base_folder = "/Users/dirac/Projects-Data/z3-haar-data" 

    today_date = Dates.format(Dates.today(), "yyyy-mm-dd")
    parent_dir = dirname(base_folder)
    folder_name = basename(base_folder) * "_" * today_date
    folder = joinpath(parent_dir, folder_name)
    mkpath(folder)    
    filename_entropy = joinpath(folder, "L$(L),T$(T),dt$(dt),p$(p),dirQ,s$(shot)_hc.npy")
    npzwrite(filename_entropy, S_list)
    """
    
    
    # Data storage for cluster
    filename_entropy = "L$(L),T$(T),dt$(dt),p$(p),dirQ,s$(shot)_se.npy"
    npzwrite(filename_entropy, SE_list)
    
    
    #println("Shannon entropy: ", shannon_entropy(s_t, L))

    return SE_list
end

Entropy_t(16, 160.0, 1.0, 0.4, 1)

#using ArgParse
"""
function main()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--L"
            arg_type = Int
            required = true
        "--p"
            arg_type = Float64
            required = true
        "--start_shot"
            arg_type = Int
            required = true
        "--end_shot"
            arg_type = Int
            required = true
    end

    parsed = parse_args(s)
    L = parsed["L"]
    p = parsed["p"]
    start_shot = parsed["start_shot"]
    end_shot = parsed["end_shot"]

    # enforce positive integers
    if start_shot < 1 || end_shot < 1
        error("Shot indices must be positive integers")
    end
    if end_shot < start_shot
        error("end_shot (end_shot) must be >= start_shot (start_shot)")
    end

    for T in [10.0*L]
        for dt in [1.0]
            for shot in start_shot:end_shot
                Entropy_t(L, T, dt, p, shot)
            end
        end
    end
end

main()
"""
