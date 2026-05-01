using Base.Threads
using Random

include("z3-ancilla.jl")

Ls     = [6, 8, 10]
ps     = collect(range(0.0, 1.0, length=21))
shots  = 1:500

jobs = [(L, p, s) for L in Ls for p in ps for s in shots]
shuffle!(jobs)  # balance heavy (large L) work across threads

println("Running $(length(jobs)) jobs on $(nthreads()) threads -> $(DATA_DIR)")
flush(stdout)

done = Atomic{Int}(0)
total = length(jobs)

@threads for job in jobs
    L, p, s = job
    T = 2.0 * L
    Entropy_t(L, T, p, s)
    n = atomic_add!(done, 1) + 1
    if n % 100 == 0 || n == total
        @printf("[%5d/%5d] L=%d p=%.3f shot=%d (thread %d)\n",
                n, total, L, p, s, threadid())
        flush(stdout)
    end
end

println("Done. Data written to $(DATA_DIR)")
