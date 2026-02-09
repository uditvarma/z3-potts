using Random
using SparseArrays
using LinearAlgebra
using Arpack
using Statistics
using Random
using DelimitedFiles
using NPZ

Random.seed!(RandomDevice())

function random_product_state(L::Int)
    dim = 3^L
    ψ = Vector{ComplexF64}(undef, dim)

    # Generate local states
    locals = Vector{NTuple{3,ComplexF64}}(undef, L)
    for i in 1:L
        θ1, θ2 = rand() * π, rand() * π
        ϕ1, ϕ2 = rand() * 2π, rand() * 2π

        locals[i] = (
            cos(θ1 / 2),
            exp(im * ϕ1) * sin(θ1 / 2) * sin(θ2 / 2),
            exp(im * ϕ2) * sin(θ1 / 2) * cos(θ2 / 2)
        )
    end

    # Fill ψ using base-3 indexing
    for idx in 0:dim-1
        x = idx
        amp = one(ComplexF64)
        for i in 1:L
            s = (x % 3) + 1
            amp *= locals[i][s]
            x ÷= 3
        end
        ψ[idx + 1] = amp
    end

    normalize!(ψ)
    return ψ
end

function superposition_state(L::Int)
    dim = 3^L
    fill(ComplexF64(1 / sqrt(dim)), dim)
end

function haar_unitary_sparse(n::Int)
    A = randn(ComplexF64, n, n)
    Q, R = qr(A)
    phases = Diagonal(R) ./ abs.(Diagonal(R))
    return sparse(Q * Diagonal(phases))
end

function z3_haar()
    U0 = haar_unitary_sparse(3)
    U1 = haar_unitary_sparse(3)
    U2 = haar_unitary_sparse(3)

    U = zeros(ComplexF64, 9, 9)
    U[1:3, 1:3] = U0
    U[4:6, 4:6] = U1
    U[7:9, 7:9] = U2

    return U
end

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

function apply_two_site_gate!(
    ψ::Vector{ComplexF64},
    U::Matrix{ComplexF64},
    L::Int,
    site::Int
)
    # ψ reshaped so sites (site, site+1) are grouped
    left  = 3^(site - 1)
    right = 3^(L - site - 1)

    ψr = reshape(ψ, left, 9, right)

    @views for l in 1:left, r in 1:right
        ψr[l, :, r] = U * ψr[l, :, r]
    end

    return ψ
end

function odd_layer!(ψ::Vector{ComplexF64}, L::Int)
    for site in 1:2:L-1
        U = transform(z3_haar())
        apply_two_site_gate!(ψ, U, L, site)
    end
    return ψ
end

function even_layer!(ψ::Vector{ComplexF64}, L::Int)
    for site in 2:2:L-1
        U = transform(z3_haar())
        apply_two_site_gate!(ψ, U, L, site)
    end
    return ψ
end

function time_evolution!(ψ::Vector{ComplexF64}, L::Int)
    odd_layer!(ψ, L)
    even_layer!(ψ, L)
    normalize!(ψ)
    return ψ
end

function calculate_entropy(psi::AbstractVector, L::Integer, subsystem_sites::AbstractVector{<:Integer}; 
    d::Integer=3)
    # Check if vector size matches the physical dimensions
    if length(psi) != d^L
        error("Dimension mismatch: Vector length $(length(psi)) does not match d^L ($d^$L = $(d^L))")
    end

    # 1. Reshape vector into a tensor with L indices of dimension d
    # usage of Tuple(fill(d, L)) ensures type stability for reshape
    psi_tensor = reshape(psi, ntuple(_ -> d, L))

    # 2. Identify environment sites
    # Using a Set for 'subsystem_sites' makes the lookup O(1) instead of O(N)
    sub_set = Set(subsystem_sites)
    environment_sites = [i for i in 1:L if i ∉ sub_set]
    
    # 3. Permute dimensions to group subsystem sites at the front
    permute_order = vcat(subsystem_sites, environment_sites)
    psi_permuted = permutedims(psi_tensor, permute_order)

    # 4. Reshape into a Matrix (dim_subsystem x dim_environment)
    dim_subsystem = d^length(subsystem_sites)
    dim_environment = d^length(environment_sites)
    psi_matrix = reshape(psi_permuted, dim_subsystem, dim_environment)

    # 5. Compute Singular Values (SVD)
    # svdvals is much faster than svd() because it doesn't compute U and V vectors
    s = svdvals(psi_matrix)

    # 6. Compute Entropy
    s_squared = s.^2
    
    # Normalize here instead of normalizing 'psi' at the start (saves 2 allocations)
    total_prob = sum(s_squared)
    s_squared ./= total_prob

    # Filter non-zero values to avoid log(0)
    # Using a slightly higher tolerance for stability
    filter!(x -> x > 1e-15, s_squared)
    
    entropy = -sum(s_squared .* log.(s_squared))

    return entropy
end

