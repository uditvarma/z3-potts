using SparseArrays
using LinearAlgebra
using Arpack
using Statistics
using Random
using DelimitedFiles
using NPZ
using ExpmV
using Dates

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
    return blockdiag(U0, U1, U2)  # built-in sparse blockdiag
end

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

function transform(U_z3)
    P = permutation_matrix_sparse()
    return P * U_z3 * P'
end

function get_z3_operators()
    id = sparse(ComplexF64[1 0 0; 0 1 0; 0 0 1])
    τ = sparse(ComplexF64[1 0 0; 0 exp(2im * pi / 3) 0; 0 0 exp(4im * pi / 3)]) ## τ in Romain's paper

    return id, τ
end

function build_term(operators::Vector{<:SparseMatrixCSC}) 
    term = operators[1]
    for j in 2:length(operators)
        term = kron(term, operators[j])
    end
    return term
end

function create_local_tau_operator(L::Int, site::Int)
    id, τ = get_z3_operators()
    ops = fill(id, L)
    ops[site] = (im / sqrt(3)) * (τ' - τ)
    return build_term(ops)
end