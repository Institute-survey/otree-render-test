#!/usr/bin/env julia
using Distributed

function print_help()
    println("""
Run expanded PGG simulations with subjective marginal-effect learning.

Options:
  --output-dir DIR          Output directory (default: expanded_pgg_outputs_marginal_learning)
  --groups N                Number of groups per condition (default: 100)
  --rounds N                Number of rounds per group (default: 100)
  --workers N               Total number of worker processes
                            (default: number of logical CPU threads)
  --seed N                  Base seed (default: 12345)
  --include-baseline        Include the baseline condition exactly once (default: on)
  --exclude-baseline        Exclude the baseline condition
  --max-conditions N        Optional cap on number of conditions, useful for smoke tests
  -h, --help                Show this help message and exit
""")
end

function extract_option_value(argv::Vector{String}, i::Int, flag::String)
    arg = argv[i]
    prefix = flag * "="
    if arg == flag
        i == length(argv) && error("Missing value after $(flag)")
        return argv[i + 1], i + 1
    elseif startswith(arg, prefix)
        return argv[length(flag) + 2:end], i
    else
        error("Unexpected parser state for $(flag)")
    end
end

function parse_args(argv::Vector{String})
    output_dir = "expanded_pgg_outputs_marginal_learning"
    groups = 100
    rounds = 100
    workers = nothing
    seed = 12345
    include_baseline = true
    exclude_baseline = false
    max_conditions = nothing

    i = 1
    while i <= length(argv)
        arg = argv[i]
        if arg == "-h" || arg == "--help"
            print_help()
            exit(0)
        elseif arg == "--include-baseline"
            include_baseline = true
        elseif arg == "--exclude-baseline"
            exclude_baseline = true
        elseif arg == "--output-dir" || startswith(arg, "--output-dir=")
            value, i = extract_option_value(argv, i, "--output-dir")
            output_dir = value
        elseif arg == "--groups" || startswith(arg, "--groups=")
            value, i = extract_option_value(argv, i, "--groups")
            groups = parse(Int, value)
        elseif arg == "--rounds" || startswith(arg, "--rounds=")
            value, i = extract_option_value(argv, i, "--rounds")
            rounds = parse(Int, value)
        elseif arg == "--workers" || startswith(arg, "--workers=")
            value, i = extract_option_value(argv, i, "--workers")
            workers = parse(Int, value)
        elseif arg == "--seed" || startswith(arg, "--seed=")
            value, i = extract_option_value(argv, i, "--seed")
            seed = parse(Int, value)
        elseif arg == "--max-conditions" || startswith(arg, "--max-conditions=")
            value, i = extract_option_value(argv, i, "--max-conditions")
            max_conditions = parse(Int, value)
        else
            error("Unknown argument: $(arg)")
        end
        i += 1
    end

    return (
        output_dir = output_dir,
        groups = groups,
        rounds = rounds,
        workers = workers,
        seed = seed,
        include_baseline = include_baseline,
        exclude_baseline = exclude_baseline,
        max_conditions = max_conditions,
    )
end

function adjust_workers!(target_workers::Int)
    target_workers >= 1 || error("--workers must be at least 1")
    current_workers = nworkers()
    delta = target_workers - current_workers
    if delta > 0
        addprocs(delta)
    elseif delta < 0
        rmprocs(workers()[1:(-delta)])
    end
    return nothing
end

const PARSED_ARGS = parse_args(ARGS)
const TARGET_WORKERS = PARSED_ARGS.workers === nothing ? max(1, Sys.CPU_THREADS) : PARSED_ARGS.workers
adjust_workers!(TARGET_WORKERS)

@everywhere begin
    using Random
    using Statistics

    const GROUP_SIZE = 4
    const BASE_ENDOWMENT = 100.0
    const INITIAL_ENDOWMENT = 100.0
    const ENDOWMENT_FLOOR = 1.0
    const MULTIPLIER = 1.6
    const MPCR = MULTIPLIER / GROUP_SIZE
    const ACTION_LEVELS = 11
    const ALPHA = 0.2

    const RHO_GRID = [0.2, 0.5, 0.8]
    const SIGMA_E_GRID = [5.0, 15.0, 30.0]
    const BASE_ERROR_PROB_GRID = [0.0, 0.1]
    const ERROR_SENSITIVITY_GRID = [0.0, 1.5, 3.0]
    const ERROR_MODE_GRID = ["flip", "random"]
    const EARNED_GRID = [false, true]
    const COST_WEIGHT_INCREASE_GRID = [0.1, 0.3]
    const GRATITUDE_STEPS_GRID = [0, 1, 5]

    struct Condition
        condition_type::String
        rho::Float64
        sigma_e::Float64
        base_error_prob::Float64
        error_sensitivity::Float64
        error_mode::String
        earned::Bool
        cost_weight_increase::Float64
        gratitude_steps::Int
    end

    struct TaskSpec
        condition::Condition
        condition_id::String
        group_id::Int
        rounds::Int
        seed::Int
    end

    format_one_decimal(value::Float64) = string(round(value; digits = 1))

    function condition_id(cond::Condition)
        if cond.condition_type == "baseline"
            return "baseline"
        end
        earned_label = cond.earned ? "earned" : "not_earned"
        cwi = cond.earned ? format_one_decimal(cond.cost_weight_increase) : "na"
        grat = cond.earned ? string(cond.gratitude_steps) : "na"
        return string(
            "rho_", format_one_decimal(cond.rho),
            "__sigma_", Int(cond.sigma_e),
            "__p0_", format_one_decimal(cond.base_error_prob),
            "__alpha_", format_one_decimal(cond.error_sensitivity),
            "__err_", cond.error_mode,
            "__", earned_label,
            "__cost_", cwi,
            "__grat_", grat,
        )
    end

    function code_from_decimal(value::Float64; scale::Int = 10, width::Int = 2)
        return lpad(string(round(Int, value * scale)), width, '0')
    end

    function condition_code(cond::Condition)
        if cond.condition_type == "baseline"
            return "base"
        end

        rho_code = code_from_decimal(cond.rho; scale = 10, width = 2)
        sigma_code = lpad(string(round(Int, cond.sigma_e)), 2, '0')
        p0_code = code_from_decimal(cond.base_error_prob; scale = 10, width = 2)
        alpha_code = code_from_decimal(cond.error_sensitivity; scale = 10, width = 2)
        err_code_map = Dict("none" => "n", "flip" => "f", "random" => "r")
        err_code = err_code_map[cond.error_mode]

        if !cond.earned
            return "r$(rho_code)_s$(sigma_code)_p$(p0_code)_a$(alpha_code)_$(err_code)_n"
        end

        cost_code = code_from_decimal(cond.cost_weight_increase; scale = 10, width = 2)
        return "r$(rho_code)_s$(sigma_code)_p$(p0_code)_a$(alpha_code)_$(err_code)_e_c$(cost_code)_g$(Int(cond.gratitude_steps))"
    end

    condition_filename(cond::Condition) = string(condition_code(cond), ".csv")

    function build_conditions(; include_baseline::Bool = true)
        conditions = Condition[]
        if include_baseline
            push!(conditions, Condition("baseline", 0.0, 0.0, 0.0, 0.0, "none", false, 0.0, 0))
        end

        for rho in RHO_GRID
            for sigma_e in SIGMA_E_GRID
                for p0 in BASE_ERROR_PROB_GRID
                    for alpha in ERROR_SENSITIVITY_GRID
                        for err_mode in ERROR_MODE_GRID
                            for earned in EARNED_GRID
                                if earned
                                    for cwi in COST_WEIGHT_INCREASE_GRID
                                        for grat in GRATITUDE_STEPS_GRID
                                            push!(conditions, Condition("experimental", rho, sigma_e, p0, alpha, err_mode, true, cwi, grat))
                                        end
                                    end
                                else
                                    push!(conditions, Condition("experimental", rho, sigma_e, p0, alpha, err_mode, false, 0.0, 0))
                                end
                            end
                        end
                    end
                end
            end
        end
        return conditions
    end

    action_grid_for_endowment(endowment::Float64) = collect(range(0.0; stop = endowment, length = ACTION_LEVELS))
    nearest_index(values::Vector{Float64}, target::Float64) = findmin(abs.(values .- target))[2]
    clip_index(index::Int) = max(1, min(ACTION_LEVELS, index))

    function generate_group_endowments(rounds::Int, rho::Float64, sigma_e::Float64, rng::AbstractRNG)
        endowments = Vector{Float64}(undef, rounds)
        endowments[1] = max(ENDOWMENT_FLOOR, INITIAL_ENDOWMENT)
        for t in 2:rounds
            shock = randn(rng) * sigma_e
            endowments[t] = rho * endowments[t - 1] + (1.0 - rho) * BASE_ENDOWMENT + shock
            if endowments[t] < ENDOWMENT_FLOOR
                endowments[t] = ENDOWMENT_FLOOR
            end
        end
        return endowments
    end

    subjective_cost_multiplier(cond::Condition) = cond.earned ? (1.0 + cond.cost_weight_increase) : 1.0
    subjective_marginal_effect(cond::Condition) = MPCR - subjective_cost_multiplier(cond)

    function simulate_group(condition::Condition, rounds::Int, seed::Int)
        rng = MersenneTwister(seed)
        endowments = generate_group_endowments(rounds, condition.rho, condition.sigma_e, rng)

        marginal_estimates = zeros(Float64, GROUP_SIZE)
        target_marginal = subjective_marginal_effect(condition)

        initial_grid = action_grid_for_endowment(endowments[1])
        current_indices = rand(rng, 1:ACTION_LEVELS, GROUP_SIZE)
        current_contributions = initial_grid[current_indices]

        mean_rates = Vector{Float64}(undef, rounds)

        for t in 1:rounds
            current_endowment = endowments[t]
            total_contribution = float(sum(current_contributions))
            share_from_others = MPCR .* (total_contribution .- current_contributions)

            mean_rates[t] = float(mean(current_contributions ./ current_endowment))
            marginal_estimates .+= ALPHA .* (target_marginal .- marginal_estimates)

            if t == rounds
                break
            end

            next_endowment = endowments[t + 1]
            next_grid = action_grid_for_endowment(next_endowment)
            next_indices = Vector{Int}(undef, GROUP_SIZE)
            next_contributions = Vector{Float64}(undef, GROUP_SIZE)

            volatility = t >= 2 ? abs(endowments[t] - endowments[t - 1]) / 100.0 : 0.0
            p_error = min(1.0, condition.base_error_prob + condition.error_sensitivity * volatility)

            for i in 1:GROUP_SIZE
                if marginal_estimates[i] > 0
                    direction = 1
                elseif marginal_estimates[i] < 0
                    direction = -1
                else
                    direction = 0
                end

                if condition.error_mode != "none" && rand(rng) < p_error
                    if condition.error_mode == "flip"
                        direction = direction != 0 ? -direction : rand(rng, (-1, 1))
                    elseif condition.error_mode == "random"
                        direction = rand(rng, (-1, 0, 1))
                    else
                        throw(ArgumentError("Unknown error_mode: $(condition.error_mode)"))
                    end
                end

                gratitude_extra_steps = 0
                if condition.earned && condition.gratitude_steps > 0 && share_from_others[i] > 0
                    gratitude_extra_steps = condition.gratitude_steps
                end

                reference_amount = current_contributions[i]
                base_index = nearest_index(next_grid, reference_amount)
                new_index = clip_index(base_index + direction + gratitude_extra_steps)

                next_indices[i] = new_index
                next_contributions[i] = next_grid[new_index]
            end

            current_indices = next_indices
            current_contributions = next_contributions
        end

        return mean_rates, endowments
    end

    function run_task(task::TaskSpec)
        mean_rates, endowments = simulate_group(task.condition, task.rounds, task.seed)
        return task.condition_id, task.group_id, mean_rates, endowments
    end

    function build_tasks(conditions::Vector{Condition}, groups::Int, rounds::Int, base_seed::Int)
        tasks = TaskSpec[]
        for (condition_index, cond) in enumerate(conditions)
            cid = condition_id(cond)
            for group_id in 1:groups
                seed = base_seed + (condition_index - 1) * 1_000_000 + group_id
                push!(tasks, TaskSpec(cond, cid, group_id, rounds, seed))
            end
        end
        return tasks
    end

    function csv_escape(value)
        s = value isa Bool ? (value ? "True" : "False") : string(value)
        if occursin(',', s) || occursin('"', s) || occursin('\n', s) || occursin('\r', s)
            return "\"" * replace(s, "\"" => "\"\"") * "\""
        end
        return s
    end

    function write_csv(path::AbstractString, header::Vector{String}, rows::Vector{Vector{Any}})
        open(path, "w") do io
            println(io, join(header, ","))
            for row in rows
                println(io, join(csv_escape.(row), ","))
            end
        end
    end

    function wide_rows(series_list::Vector{Vector{Float64}}, rounds::Int)
        rows = Vector{Vector{Any}}()
        for (group_index, series) in enumerate(series_list)
            row = Any[group_index]
            for r in 1:rounds
                push!(row, Float64(series[r]))
            end
            push!(rows, row)
        end
        return rows
    end

    function add_metadata_rows(df_rows::Vector{Vector{Any}}, meta::Dict{String,Any})
        out_rows = Vector{Vector{Any}}()
        for row in df_rows
            base = Any[
                meta["condition_id"],
                meta["condition_code"],
                meta["filename"],
                meta["condition_type"],
                meta["rho"],
                meta["sigma_e"],
                meta["base_error_prob"],
                meta["error_sensitivity"],
                meta["error_mode"],
                meta["earned"],
                meta["cost_weight_increase"],
                meta["gratitude_steps"],
            ]
            append!(base, row)
            push!(out_rows, base)
        end
        return out_rows
    end

    function write_outputs(output_dir::AbstractString,
                           conditions::Vector{Condition},
                           cooperation_by_condition::Dict{String,Vector{Vector{Float64}}},
                           endowment_by_condition::Dict{String,Vector{Vector{Float64}}},
                           rounds::Int)
        condition_dir = joinpath(output_dir, "condition_csvs")
        endowment_dir = joinpath(output_dir, "endowment_csvs")
        mkpath(condition_dir)
        mkpath(endowment_dir)

        manifest_rows = Vector{Vector{Any}}()
        combined_rows = Vector{Vector{Any}}()
        endowment_combined_rows = Vector{Vector{Any}}()

        rounds_header = ["round_$(i)" for i in 1:rounds]
        wide_header = vcat(["group_id"], rounds_header)
        combined_header = vcat([
            "condition_id",
            "condition_code",
            "filename",
            "condition_type",
            "rho",
            "sigma_e",
            "base_error_prob",
            "error_sensitivity",
            "error_mode",
            "earned",
            "cost_weight_increase",
            "gratitude_steps",
            "group_id",
        ], rounds_header)

        for cond in conditions
            cid = condition_id(cond)
            ccode = condition_code(cond)
            filename = condition_filename(cond)

            coop_rows = wide_rows(cooperation_by_condition[cid], rounds)
            endowment_rows = wide_rows(endowment_by_condition[cid], rounds)

            coop_csv_path = joinpath(condition_dir, filename)
            endowment_csv_path = joinpath(endowment_dir, filename)
            write_csv(coop_csv_path, wide_header, coop_rows)
            write_csv(endowment_csv_path, wide_header, endowment_rows)

            meta = Dict{String,Any}(
                "condition_type" => cond.condition_type,
                "rho" => cond.rho,
                "sigma_e" => cond.sigma_e,
                "base_error_prob" => cond.base_error_prob,
                "error_sensitivity" => cond.error_sensitivity,
                "error_mode" => cond.error_mode,
                "earned" => cond.earned,
                "cost_weight_increase" => cond.cost_weight_increase,
                "gratitude_steps" => cond.gratitude_steps,
                "condition_id" => cid,
                "condition_code" => ccode,
                "filename" => filename,
                "cooperation_csv_path" => coop_csv_path,
                "endowment_csv_path" => endowment_csv_path,
            )

            push!(manifest_rows, Any[
                meta["condition_type"],
                meta["rho"],
                meta["sigma_e"],
                meta["base_error_prob"],
                meta["error_sensitivity"],
                meta["error_mode"],
                meta["earned"],
                meta["cost_weight_increase"],
                meta["gratitude_steps"],
                meta["condition_id"],
                meta["condition_code"],
                meta["filename"],
                meta["cooperation_csv_path"],
                meta["endowment_csv_path"],
            ])

            append!(combined_rows, add_metadata_rows(coop_rows, meta))
            append!(endowment_combined_rows, add_metadata_rows(endowment_rows, meta))
        end

        sort!(manifest_rows, by = row -> (row[1], row[11], row[10]))
        manifest_header = [
            "condition_type",
            "rho",
            "sigma_e",
            "base_error_prob",
            "error_sensitivity",
            "error_mode",
            "earned",
            "cost_weight_increase",
            "gratitude_steps",
            "condition_id",
            "condition_code",
            "filename",
            "cooperation_csv_path",
            "endowment_csv_path",
        ]
        write_csv(joinpath(output_dir, "manifest_conditions.csv"), manifest_header, manifest_rows)

        sort!(combined_rows, by = row -> (row[4], row[2], row[13]))
        write_csv(joinpath(output_dir, "all_conditions_group_means.csv"), combined_header, combined_rows)

        sort!(endowment_combined_rows, by = row -> (row[4], row[2], row[13]))
        write_csv(joinpath(output_dir, "all_conditions_endowments.csv"), combined_header, endowment_combined_rows)
    end
end

function run_all_conditions(; output_dir::AbstractString,
                             groups::Int,
                             rounds::Int,
                             seed::Int,
                             include_baseline::Bool,
                             max_conditions::Union{Nothing,Int})
    mkpath(output_dir)

    conditions = build_conditions(include_baseline = include_baseline)
    if max_conditions !== nothing
        conditions = conditions[1:min(length(conditions), max_conditions)]
    end

    tasks = build_tasks(conditions, groups, rounds, seed)
    isempty(tasks) && error("No tasks were generated. Check condition settings.")

    cooperation_by_condition = Dict{String,Vector{Vector{Float64}}}()
    endowment_by_condition = Dict{String,Vector{Vector{Float64}}}()
    for cond in conditions
        cid = condition_id(cond)
        cooperation_by_condition[cid] = [Float64[] for _ in 1:groups]
        endowment_by_condition[cid] = [Float64[] for _ in 1:groups]
    end

    completed = Ref(0)
    total = length(tasks)
    progress_lock = ReentrantLock()

    function record_result!(cid::String, group_id::Int, mean_rates::Vector{Float64}, endowments::Vector{Float64})
        lock(progress_lock) do
            cooperation_by_condition[cid][group_id] = mean_rates
            endowment_by_condition[cid][group_id] = endowments
            completed[] += 1
            if completed[] % 1000 == 0 || completed[] == total
                println("Completed $(completed[])/$(total) simulations")
            end
        end
        return nothing
    end

    if isempty(Distributed.workers())
        for task in tasks
            cid, group_id, mean_rates, endowments = run_task(task)
            record_result!(cid, group_id, mean_rates, endowments)
        end
    else
        task_channel = Channel{TaskSpec}(length(tasks))
        for task in tasks
            put!(task_channel, task)
        end
        close(task_channel)

        @sync for pid in Distributed.workers()
            @async begin
                for task in task_channel
                    cid, group_id, mean_rates, endowments = remotecall_fetch(run_task, pid, task)
                    record_result!(cid, group_id, mean_rates, endowments)
                end
            end
        end
    end

    write_outputs(output_dir, conditions, cooperation_by_condition, endowment_by_condition, rounds)

    println("Saved manifest to: $(joinpath(output_dir, \"manifest_conditions.csv\"))")
    println("Saved cooperation results to: $(joinpath(output_dir, \"all_conditions_group_means.csv\"))")
    println("Saved Endowment results to: $(joinpath(output_dir, \"all_conditions_endowments.csv\"))")
    println("Per-condition cooperation CSVs are in: $(joinpath(output_dir, \"condition_csvs\"))")
    println("Per-condition Endowment CSVs are in: $(joinpath(output_dir, \"endowment_csvs\"))")
    return nothing
end

function main(args)
    include_baseline = args.include_baseline && !args.exclude_baseline
    run_all_conditions(
        output_dir = args.output_dir,
        groups = args.groups,
        rounds = args.rounds,
        seed = args.seed,
        include_baseline = include_baseline,
        max_conditions = args.max_conditions,
    )
end

main(PARSED_ARGS)
