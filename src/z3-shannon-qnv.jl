using LinearAlgebra
using Random
using StatsBase
using NPZ
using Printf
using Dates
#using ArgParse

const QNV_OUTDIR = "z3-haar-qnv-$(Dates.format(Dates.today(), "yyyy-mm-dd"))"

# --- Helper Functions ---

function superposition_state(L::Int)
    psi = ones(ComplexF64, 3^L)
    return psi ./ norm(psi)
end

function state_one(L::Int)
    psi = zeros(ComplexF64, 3^L)
    
    # Calculate the integer value where every base-3 digit is 1
    # Example for L=3: 1*3^0 + 1*3^1 + 1*3^2 = 1 + 3 + 9 = 13.
    # Formula: (3^L - 1) / 2
    idx = div(3^L - 1, 2) + 1 
    
    psi[idx] = 1.0 + 0.0im # Set only this specific basis state to 1
    return psi
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

# Variance of the conserved Z_3 charge Q = (sum_i q_i) mod 3, treated as Q in {0,1,2}.
# Below the sharpening transition P(Q) is ~uniform => Var -> 2/3.
# Above, P(Q) sharpens onto a single sector => Var -> 0.
function charge_variance(psi, L)
    probs = zeros(3)
    for i in 0:(3^L - 1)
        amp = abs2(psi[i+1])
        if amp > 1e-15
            v, q = i, 0
            for _ in 1:L; q += v % 3; v ÷= 3; end
            probs[(q % 3) + 1] += amp
        end
    end
    mean_Q  = probs[2] + 2*probs[3]
    mean_Q2 = probs[2] + 4*probs[3]
    #return mean_Q
    return mean_Q2 - mean_Q^2
end

# --- Main Function ---

function ChargeVar_t(L::Int, T::Float64, p::Float64, shot::Int)

    s_t = superposition_state(L)
    #s_t = state_one(L)

    QV_list = Float64[]
    push!(QV_list, charge_variance(s_t, L)) # t=0

    steps = Int(floor(T))

    for _ in 1:steps

        # 1. Time Evolution
        for x in 1:2:(L-1)
            apply_gate!(s_t, construct_gate(), x, L)
        end
          
        if p > 0
            for l in 1:L
                if rand() < p
                    measure_site!(s_t, l, L)
                end
            end
        end

        for x in 2:2:(L-1)
            apply_gate!(s_t, construct_gate(), x, L)
        end

        # 2. Measurements
        if p > 0
            for l in 1:L
                if rand() < p
                    measure_site!(s_t, l, L)
                end
            end
        end

        # 3. Record charge variance
        push!(QV_list, charge_variance(s_t, L))
    end

    isdir(QNV_OUTDIR) || mkpath(QNV_OUTDIR)
    filename = joinpath(QNV_OUTDIR, "L$(L),T$(T),p$(p),s$(shot)_qv.npy")
    npzwrite(filename, QV_list)

    return QV_list
end

#ChargeVar_t(20, 40.0, 0.7, 1)

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

    if start_shot < 1 || end_shot < 1
        error("Shot indices must be positive integers")
    end
    if end_shot < start_shot
        error("end_shot (end_shot) must be >= start_shot (start_shot)")
    end

    for T in [2.0*L]
        for shot in start_shot:end_shot
            ChargeVar_t(L, T, p, shot)
        end
    end
end

main()
"""
