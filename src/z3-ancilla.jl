using LinearAlgebra
using Random
using StatsBase
using Statistics
using NPZ
#using ArgParse
using Printf
using Dates

const DATA_DIR = "z3-ancilla-mipt-data-$(Dates.format(Dates.today(), "yyyy-mm-dd"))"
isdir(DATA_DIR) || mkpath(DATA_DIR)

# ==========================================
# 1. PHYSICS HELPER FUNCTIONS
# ==========================================

function initial_mipt(L::Int)
    @assert L >= 2 "System size must be at least 2"
    
    # State matrix: (System_Dim, Ancilla_Dim=3)
    psi_total = zeros(ComplexF64, 3^L, 3)
    
    # Branch 1: |00...00>_S ⊗ |0>_A  (Column 1)
    # All sites are 0, base-3 integer is 0.
    idx_1 = 0
    psi_total[idx_1 + 1, 1] = 1.0 / sqrt(3)
    
    # Branch 2: |00...↑↓>_S ⊗ |↑>_A  (Column 2)
    # ↑ (1) at L-1, ↓ (2) at L
    idx_2 = 1 * 3^(L-2) + 2 * 3^(L-1)
    psi_total[idx_2 + 1, 2] = 1.0 / sqrt(3)
    
    # Branch 3: |00...↓↑>_S ⊗ |↓>_A  (Column 3)
    # ↓ (2) at L-1, ↑ (1) at L
    idx_3 = 2 * 3^(L-2) + 1 * 3^(L-1)
    psi_total[idx_3 + 1, 3] = 1.0 / sqrt(3)
    
    return psi_total
end

function initial_entangled_state_ghz(L::Int)
    @assert L >= 2 "System size must be at least 2"
    
    # State matrix: (System_Dim, Ancilla_Dim=3)
    psi_total = zeros(ComplexF64, 3^L, 3)
    
    # Branch 1: |00...00>_S ⊗ |0>_A
    idx_1 = 0
    psi_total[idx_1 + 1, 1] = 1.0 / sqrt(3)
    
    # Branch 2: |↑↑...↑↑>_S ⊗ |↑>_A 
    idx_2 = div(3^L - 1, 2)
    psi_total[idx_2 + 1, 2] = 1.0 / sqrt(3)
    
    # Branch 3: |↓↓...↓↓>_S ⊗ |↓>_A 
    idx_3 = 3^L - 1
    psi_total[idx_3 + 1, 3] = 1.0 / sqrt(3)
    
    return psi_total
end

function construct_gate()
    function haar()
        Z = randn(ComplexF64, 3, 3)
        Q, R = qr(Z)
        return Q * Diagonal(diag(R) ./ abs.(diag(R)))
    end
    
    U = zeros(ComplexF64, 9, 9)
    # Block diagonal structure
    idx0 = [1, 6, 8]; idx1 = [2, 4, 9]; idx2 = [3, 5, 7]
    U[idx0, idx0] = haar(); U[idx1, idx1] = haar(); U[idx2, idx2] = haar()
    return U
end

# Apply Gate (Handles Matrix State)
function apply_gate!(psi::Matrix{ComplexF64}, U::Matrix{ComplexF64}, site::Int, L::Int)
    dL, dR = 3^(site-1), 3^(L-site-1)
    for a in 1:3
        psi_col = @view psi[:, a]
        psi_t = permutedims(reshape(psi_col, dL, 3, 3, dR), (2, 3, 1, 4))
        psi_new = U * reshape(psi_t, 9, :)
        psi[:, a] = reshape(permutedims(reshape(psi_new, 3, 3, dL, dR), (3, 1, 2, 4)), :)
    end
end

# Apply PBC Boundary Gate
function apply_boundary_gate!(psi::Matrix{ComplexF64}, U::Matrix{ComplexF64}, L::Int)
    for a in 1:3
        psi_col = @view psi[:, a]
        psi_reshaped = reshape(psi_col, 3, 3^(L-2), 3)
        psi_perm = permutedims(psi_reshaped, (3, 1, 2)) # (L, 1, Middle)
        psi_mat = reshape(psi_perm, 9, :)
        psi_new_mat = U * psi_mat
        psi_new_tensor = reshape(psi_new_mat, 3, 3, :)
        psi_final = permutedims(psi_new_tensor, (2, 3, 1)) # (1, Middle, L)
        psi[:, a] = reshape(psi_final, :)
    end
end

# Measurement
function measure_site!(psi::Matrix{ComplexF64}, site::Int, L::Int)
    dL, dR = 3^(site-1), 3^(L-site)
    probs = zeros(Float64, 3)
    
    # Sum probabilities from all 3 branches
    for a in 1:3
        psi_view = reshape(@view(psi[:, a]), dL, 3, dR)
        for s in 1:3
            probs[s] += sum(abs2, psi_view[:, s, :])
        end
    end
    
    if sum(probs) > 1e-15
        outcome = sample(1:3, Weights(probs))
        norm_fac = 1.0 / sqrt(probs[outcome])
        
        # Collapse all branches consistently
        for a in 1:3
            psi_view = reshape(@view(psi[:, a]), dL, 3, dR)
            for s in 1:3
                if s != outcome
                    psi_view[:, s, :] .= 0
                else
                    psi_view[:, s, :] .*= norm_fac
                end
            end
        end
    end
end

# Ancilla Entropy
function ancilla_entropy(psi::Matrix{ComplexF64})
    rho_A = psi' * psi 
    evals = eigen(Hermitian(rho_A)).values
    evals = max.(evals, 1e-20)
    evals ./= sum(evals)
    return -sum(p -> p * log(p), evals)
end

# ==========================================
# 2. CLUSTER EXECUTION FUNCTION
# ==========================================

function Entropy_t(L::Int, T::Float64, p::Float64, shot::Int)
    
    # Initialize state
    psi_total = initial_mipt(L)
    
    # Data storage
    SE_list = Float64[]
    push!(SE_list, ancilla_entropy(psi_total)) # t=0

    # ==========================================
    # PHASE 1: Scrambling (t = 1 to L)
    # No measurements, allows entanglement to build
    # ==========================================
    for _ in 1:L
        # Even Layer
        for x in 1:2:(L-1); apply_gate!(psi_total, construct_gate(), x, L); end
        # Odd Layer
        for x in 2:2:(L-2); apply_gate!(psi_total, construct_gate(), x, L); end
        # PBC Boundary
        apply_boundary_gate!(psi_total, construct_gate(), L)
        
        # Record Entropy
        push!(SE_list, ancilla_entropy(psi_total))
    end

    # ==========================================
    # PHASE 2: Dynamics (t = 1 to T)
    # With measurements
    # ==========================================
    steps = Int(floor(T))

    for _ in 1:steps
        # Even Layer
        for x in 1:2:(L-1); apply_gate!(psi_total, construct_gate(), x, L); end

         if p > 0
            for l in 1:L 
                if rand() < p
                    measure_site!(psi_total, l, L)
                end
            end
        end
        
        # Odd Layer
        for x in 2:2:(L-2); apply_gate!(psi_total, construct_gate(), x, L); end
        # PBC Boundary
        apply_boundary_gate!(psi_total, construct_gate(), L)

        # Measurements
        if p > 0
            for l in 1:L 
                if rand() < p
                    measure_site!(psi_total, l, L)
                end
            end
        end

        # Record Entropy
        push!(SE_list, ancilla_entropy(psi_total))
    end

    # Data storage for cluster
    # Format: L{L},T{T},p{p},s{shot}_ancilla_ent.npy
    filename_entropy = joinpath(DATA_DIR, "L$(L),T$(T),p$(p),s$(shot)_a.npy")
    npzwrite(filename_entropy, SE_list)
    
    return SE_list
end

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
