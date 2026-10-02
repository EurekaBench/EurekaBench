# Infinite-n ideal ballooning boundary of gEQDSK files with GPEC, after the TokaMaker
# NT_Ballooning_GPEC example. Writes ballooning_boundary_<case>.json beside each file.
# Usage: julia --project=$GPEC_PROJECT gpec_ballooning.jl [--both] [--n_scan=N] [--mpsi=N]
#        [--mtheta=N] [--max_alpha_scale=X] <geqdsk> [...]
# max_alpha_scale bounds the search for the boundaries, in units of the equilibrium alpha

using GeneralizedPerturbedEquilibrium
using GeneralizedPerturbedEquilibrium: Equilibrium, LocalStability

function make_eq_dict(geqdsk_path, mpsi, mtheta)
    return Dict{String,Any}(
        "eq_filename" => geqdsk_path,
        "eq_type" => "efit",
        "jac_type" => "hamada",
        "grid_type" => "log_asymptotic",
        "psilow" => 1e-4,
        "psihigh" => 0.9995,
        "mpsi" => mpsi,
        "psi_accuracy" => 0.001,
        "mtheta" => mtheta,
        "newq0" => 0,
        "etol" => 1e-10,
    )
end

function option(name, default)
    prefix = "--" * name * "="
    for arg in ARGS
        startswith(arg, prefix) && return parse(Int, arg[length(prefix)+1:end])
    end
    return default
end

function foption(name, default)
    prefix = "--" * name * "="
    for arg in ARGS
        startswith(arg, prefix) && return parse(Float64, arg[length(prefix)+1:end])
    end
    return default
end

fmt(v) = isfinite(v) ? repr(v) : "null"

function write_json(path, psi, alpha, alpha_critical, alpha_critical2)
    open(path, "w") do f
        write(f, "{\n")
        write(f, "  \"psi\": [", join(fmt.(psi), ", "), "],\n")
        write(f, "  \"alpha\": [", join(fmt.(alpha), ", "), "],\n")
        write(f, "  \"alpha_critical\": [", join(fmt.(alpha_critical), ", "), "]")
        if alpha_critical2 !== nothing
            write(f, ",\n  \"alpha_critical2\": [", join(fmt.(alpha_critical2), ", "), "]\n")
        else
            write(f, "\n")
        end
        write(f, "}\n")
    end
end

both_boundaries = "--both" in ARGS
n_scan = option("n_scan", 64)
mpsi = option("mpsi", 0)
mtheta = option("mtheta", 256)
max_alpha_scale = foption("max_alpha_scale", 8.0)
geqdsk_files = filter(arg -> !startswith(arg, "--"), ARGS)
isempty(geqdsk_files) && error("Usage: julia --project=<GPEC repo> gpec_ballooning.jl [--both] [--n_scan=N] [--mpsi=N] [--mtheta=N] [--max_alpha_scale=X] <geqdsk> [...]")

for geqdsk_file in geqdsk_files
    geqdsk_path = abspath(geqdsk_file)
    isfile(geqdsk_path) || error("File not found: $geqdsk_path")
    case_label = splitext(basename(geqdsk_path))[1]
    println("Ballooning boundary: $case_label")

    eq_config = Equilibrium.EquilibriumConfig(make_eq_dict(geqdsk_path, mpsi, mtheta), dirname(geqdsk_path))
    equil = Equilibrium.setup_equilibrium(eq_config)

    if both_boundaries
        bnd = LocalStability.ballooning_alpha_boundaries(equil; n_scan=n_scan, max_alpha_scale=max_alpha_scale)
        psi, alpha = bnd.psi, bnd.alpha
        alpha_critical, alpha_critical2 = bnd.alpha_critical1, bnd.alpha_critical2
    else
        bnd = LocalStability.ballooning_alpha_boundary(equil; n_scan=n_scan)
        psi, alpha = bnd.psi, bnd.alpha
        alpha_critical, alpha_critical2 = bnd.alpha_critical, nothing
    end

    json_path = joinpath(dirname(geqdsk_path), "ballooning_boundary_$(case_label).json")
    write_json(json_path, psi, alpha, alpha_critical, alpha_critical2)
    println("  saved: $json_path  ($(length(psi)) surfaces)")
end
