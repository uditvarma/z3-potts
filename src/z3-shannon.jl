using LinearAlgebra
using Random
using StatsBase
using NPZ
using Printf
using ArgParse

# --- Helper Functions ---

function superposition_state(L::Int)
    psi = ones(ComplexF64, 3^L)
    return psi ./ norm(psi)
end

function construct_gate()
    # Haar random 3x3 block generator
    function haar()
        Z = randn(ComplexF64, 3, 3)
        Q, R = qr(Z)
        return Q * Diagonal(diag(R) ./ abs.(diag(R)))
    end
    
    U = zeros(ComplexF64, 9, 9)
    idx0 = [1, 6, 8]; idx1 = [2, 4, 9]; idx2 = [3, 5, 7]
    U[idx0, idx0] = haar(); U[idx1, idx1] = haar(); U[idx2, idx2] = haar()
    return U
end

function apply_gate!(psi, U, site, L)
    dL, dR = 3^(site-1), 3^(L-site-1)
    # Reshape -> Permute -> Multiply -> Permute -> Reshape
    psi_t = permutedims(reshape(psi, dL, 3, 3, dR), (2, 3, 1, 4))
    psi_new = U * reshape(psi_t, 9, :)
    psi[:] = reshape(permutedims(reshape(psi_new, 3, 3, dL, dR), (3, 1, 2, 4)), :)
end

function measure_site!(psi, site, L)
    dL, dR = 3^(site-1), 3^(L-site)
    psi_view = reshape(psi, dL, 3, dR)
    
    probs = [sum(abs2, psi_view[:, s, :]) for s in 1:3]
    
    if sum(probs) > 1e-15
        outcome = sample(1:3, Weights(probs))
        norm_fac = 1.0 / sqrt(probs[outcome])
        
        for s in 1:3
            if s != outcome
                psi_view[:, s, :] .= 0
            else
                psi_view[:, s, :] .*= norm_fac
            end
        end
    end
end

function shannon_entropy(psi, L)
    probs = zeros(3)
    for i in 0:(3^L - 1)
        amp = abs2(psi[i+1])
        if amp > 1e-15
            v, q = i, 0
            for _ in 1:L; q += v % 3; v ÷= 3; end
            probs[(q % 3) + 1] += amp
        end
    end
    return -sum(p -> p > 1e-15 ? p * log(p) : 0.0, probs)
end

# --- Main Function ---

function Entropy_t(L::Int, T::Float64, p::Float64, shot::Int)
    
    # Initialize state
    s_t = superposition_state(L)
    
    # Data storage
    SE_list = Float64[]
    push!(SE_list, shannon_entropy(s_t, L)) # t=0

    steps = Int(floor(T))

    for _ in 1:steps
        
        # 1. Time Evolution
        # Even Layer
        for x in 1:2:(L-1)
            apply_gate!(s_t, construct_gate(), x, L)
        end
        # Odd Layer
        for x in 2:2:(L-1)
            apply_gate!(s_t, construct_gate(), x, L)
        end

        # 2. Measurements
        if p > 0
            # Note: Changed to 1:L to measure all sites. 
            # Use 2:L-1 if you specifically need to avoid edges.
            for l in 1:L 
                if rand() < p
                    measure_site!(s_t, l, L)
                end
            end
        end

        # 3. Record Entropy
        push!(SE_list, shannon_entropy(s_t, L))
    end

    # Data storage for cluster
    filename_entropy = "L$(L),T$(T),p$(p),s$(shot)_se.npy"
    npzwrite(filename_entropy, SE_list)
    
    return SE_list
end

#Entropy_t(20, 40.0, 0.7, 1)

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

    for T in [2.0*L]
        for shot in start_shot:end_shot
            Entropy_t(L, T, p, shot)
        end
    end
end

main()
"""