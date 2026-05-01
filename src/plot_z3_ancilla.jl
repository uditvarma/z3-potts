using NPZ
using Statistics
using Plots
using Printf
using Dates

# ==========================================
# CONFIGURATION
# ==========================================
data_dir = "z3-ancilla-mipt-data-$(Dates.format(Dates.today(), "yyyy-mm-dd"))"
# Or hard-code a previous run's folder, e.g.:
# data_dir = "z3-ancilla-mipt-data-2026-04-29"

L_values  = [6, 8, 10]
num_shots = 500
p_values  = range(0.0, 1.0, length=21)

summary_data = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}}}()

# ==========================================
# ANALYSIS LOOP
# ==========================================

for L in L_values
    println("\nProcessing System Size L = $L ...")

    T  = 2.0 * L
    dt = 1.0

    expected_length = Int(2 * T / dt) + 1     # 1 + L scramble + 2L dynamics  (matches 3L+1 only when T=L; here SE_list length is 3L+1)
    target_index    = 2L + 1                  # L dynamics steps after scrambling

    se_at_t_equals_L = Float64[]

    for p in p_values
        p_clean = round(p, digits=4)
        p_str   = string(p_clean)

        accumulated_se = Float64[]
        count_valid    = 0

        for s in 1:num_shots
            fname = "L$(L),T$(T),p$(p_str),s$(s)_a.npy"
            fpath = joinpath(data_dir, fname)

            if isfile(fpath)
                try
                    traj = npzread(fpath)
                    if isempty(accumulated_se)
                        accumulated_se = zeros(Float64, length(traj))
                    end
                    len = min(length(traj), length(accumulated_se))
                    accumulated_se[1:len] .+= traj[1:len]
                    count_valid += 1
                catch
                    # Skip corrupted
                end
            end
        end

        if count_valid > 0
            avg_traj = accumulated_se ./ count_valid
            if length(avg_traj) >= target_index
                push!(se_at_t_equals_L, avg_traj[target_index])
            else
                push!(se_at_t_equals_L, NaN)
            end
            print(".")
        else
            push!(se_at_t_equals_L, NaN)
        end
    end

    println(" Done.")
    summary_data[L] = (collect(p_values), se_at_t_equals_L)
end

# ==========================================
# SUMMARY PLOT
# ==========================================

println("\nGenerating Summary Plot...")

plt_summary = plot(
    title  = "Ancilla Entropy at t = L vs Measurement Probability",
    xlabel = "Measurement Probability p",
    ylabel = "Ancilla Entropy S(t=L)",
    legend = :topright,
    grid   = true,
    lw     = 2
)

for L in sort(collect(keys(summary_data)))
    (p_vals, se_vals) = summary_data[L]
    mask = .!isnan.(se_vals)
    plot!(plt_summary, p_vals[mask], se_vals[mask],
        label      = "L = $L",
        marker     = :circle,
        markersize = 5,
        lw         = 2
    )
end

display(plt_summary)
mkpath("../plots")
outfile = "../plots/Ancilla_at_t_equals_L_Summary_pm.pdf"
savefig(plt_summary, outfile)
println("Plot saved as $outfile")
