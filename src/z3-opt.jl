using LinearAlgebra
using Random
using Statistics
using NPZ
using Dates

Random.seed!(Dates.now().instant.periods.value)

# Creates a superposition state with supoort over all charge sectors

function superposition_state(L::Int)
    φ = ones(ComplexF64, 3) / sqrt(3)
    ψ = φ
    for _ in 2:L
        ψ = kron(ψ, φ)
    end
    return ψ
end

# Creates a Haar-random two-qubit unitary

function haar_unitary(n::Int)
    A = randn(ComplexF64, n, n)
    Q, R = qr(A)
    phases = diag(R) ./ abs.(diag(R))
    return Q * Diagonal(phases)
end

# Makes a Z3-symmetric 2-site Haar unitary

function z3_haar()
    U0 = haar_unitary(3)
    U1 = haar_unitary(3)
    U2 = haar_unitary(3)

    U = zeros(ComplexF64, 9, 9)
    U[1:3, 1:3] = U0
    U[4:6, 4:6] = U1
    U[7:9, 7:9] = U2

    return U
end

# Creates a permutation matrix that takes operators from Z3 basis to computational basis

function permutation_matrix()
    z3  = ["00","12","21","01","10","22","02","20","11"]
    std = ["00","01","02","10","11","12","20","21","22"]

    P = zeros(ComplexF64, 9, 9)
    for (i, s) in enumerate(z3)
        j = findfirst(==(s), std)
        P[j, i] = 1
    end
    return P
end

const PZ3 = permutation_matrix()
transform(U) = PZ3 * U * PZ3'

# Two-site gate applier

function apply_two_site_gate!(ψ, U, i, L)
    dimL = 3^(i-1)
    dimR = 3^(L-i-1)

    ψview = reshape(ψ, dimL, 3, 3, dimR)

    @inbounds for l in 1:dimL, r in 1:dimR
        block = reshape(@view(ψview[l, :, :, r]), 9)
        ψview[l, :, :, r] .= reshape(U * block, 3, 3)
    end

    return ψ
end


random_odd_layer(L)  = [transform(z3_haar()) for _ in 1:2:L-1]
random_even_layer(L) = [transform(z3_haar()) for _ in 2:2:L-1]

# Time evolution

function time_evolution!(ψ, L)
    # Odd layer
    Uodd = random_odd_layer(L)
    for (k, i) in enumerate(1:2:L-1)
        apply_two_site_gate!(ψ, Uodd[k], i, L)
    end

    # Even layer
    Ueven = random_even_layer(L)
    for (k, i) in enumerate(2:2:L-1)
        apply_two_site_gate!(ψ, Ueven[k], i, L)
    end

    normalize!(ψ)
end