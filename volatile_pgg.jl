using Distributed

function print_help()
    println("""
Run expanded PGG simulations with a single deduplicated baseline.

Options:
  --output-dir DIR          Output directory (default: expanded_pgg_outputs_baseline_dedup)
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
        return arg[length(prefix) + 1:end], i
    else
        error("Unexpected parser state for $(flag)")
    end
end

function parse_args(argv::Vector{String})
    output_dir = "expanded_pgg_outputs_baseline_dedup"
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
    using Printf

    const GROUP_SIZE = 4
    const BASE_ENDOWMENT = 100.0
    const INITIAL_ENDOWMENT = 100.0
    const ENDOWMENT_FLOOR = 1.0
    const MULTIPLIER = 1.6
    const MPCR = MULTIPLIER / GROUP_SIZE
    const ACTION_LEVELS = 11

    const RHO_GRID = [0.2, 0.5, 0.8]
    const SIGMA_E_GRID = [5.0, 15.0, 30.0]
    const BASE_ERROR_PROB_GRID = [0.0, 0.1]
    const ERROR_SENSITIVITY_GRID = [0.0, 1.5, 3.0]
    const ERROR_MODE_GRID = ["flip", "random"]
    const EARNED_GRID = [false, true]
    const COST_WEIGHT_INCREASE_GRID = [0.1, 0.3]
    const GRATITUDE_STEPS_GRID = [0, 1, 5]
    const DIRECTIONS = [-1, 1]
    const ERROR_CODE_MAP = Dict("none" => "n", "flip" => "f", "random" => "r")

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

    struct ManifestRow
        condition_type::String
        rho::Float64
        sigma_e::Float64
        base_error_prob::Float64
        error_sensitivity::Float64
        error_mode::String
        earned::Bool
        cost_weight_increase::Float64
        gratitude_steps::Int
        condition_id::String
        condition_code::String
        filename::String
        csv_path::String
    end

    struct CombinedRow
        condition_id::String
        condition_code::String
        filename::String
        condition_type::String
        rho::Float64
        sigma_e::Float64
        base_error_prob::Float64
        error_sensitivity::Float64
        error_mode::String
        earned::Bool
        cost_weight_increase::Float64
        gratitude_steps::Int
        group_id::Int
        series::Vector{Float64}
    end

    function condition_id(cond::Condition)
        if cond.condition_type == "baseline"
            return "baseline"
        end
        earned_label = cond.earned ? "earned" : "not_earned"
        cwi = cond.earned ? @sprintf("%.1f", cond.cost_weight_increase) : "na"
        grat = cond.earned ? string(cond.gratitude_steps) : "na"
        return string(
            "rho_", @sprintf("%.1f", cond.rho),
            "__sigma_", Int(cond.sigma_e),
            "__p0_", @sprintf("%.1f", cond.base_error_prob),
            "__alpha_", @sprintf("%.1f", cond.error_sensitivity),
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
        err_code = ERROR_CODE_MAP[cond.error_mode]

        if !cond.earned
            return "r$(rho_code)_s$(sigma_code)_p$(p0_code)_a$(alpha_code)_$(err_code)_n"
        end

        cost_code = code_from_decimal(cond.cost_weight_increase; scale = 10, width = 2)
        grat_code = string(cond.gratitude_steps)
        return "r$(rho_code)_s$(sigma_code)_p$(p0_code)_a$(alpha_code)_$(err_code)_e_c$(cost_code)_g$(grat_code)"
    end

    condition_filename(cond::Condition) = condition_code(cond) * ".csv"

    function build_conditions(; include_baseline::Bool = true)
        conditions = Condition[]

        if include_baseline
            push!(conditions, Condition(
                "baseline",
                0.0,
                0.0,
                0.0,
                0.0,
                "none",
                false,
                0.0,
                0,
            ))
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
                                            push!(conditions, Condition(
                                                "experimental",
                                                rho,
                                                sigma_e,
                                                p0,
                                                alpha,
                                                err_mode,
                                                true,
                                                cwi,
                                                grat,
                                            ))
                                        end
                                    end
                                else
                                    push!(conditions, Condition(
                                        "experimental",
                                        rho,
                                        sigma_e,
                                        p0,
                                        alpha,
                                        err_mode,
                                        false,
                                        0.0,
                                        0,
                                    ))
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

    clip_index(index::Int) = clamp(index, 1, ACTION_LEVELS)

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

    function simulate_group(condition::Condition, rounds::Int, seed::Int)
        rng = MersenneTwister(seed)

        endowments = generate_group_endowments(rounds, condition.rho, condition.sigma_e, rng)

        prev_direction = rand(rng, DIRECTIONS, GROUP_SIZE)
        prev_subjective_payoff = fill(NaN, GROUP_SIZE)

        initial_grid = action_grid_for_endowment(endowments[1])
        initial_indices = rand(rng, 1:ACTION_LEVELS, GROUP_SIZE)
        current_contributions = initial_grid[initial_indices]

        mean_rates = Vector{Float64}(undef, rounds)

        for t in 1:rounds
            current_endowment = endowments[t]
            total_contribution = sum(current_contributions)
            public_share_each = MPCR * total_contribution
            share_from_others = MPCR .* (total_contribution .- current_contributions)

            objective_payoff = current_endowment .- current_contributions .+ public_share_each
            subjective_cost_multiplier = condition.earned ? (1.0 + condition.cost_weight_increase) : 1.0
            subjective_payoff = current_endowment .- (subjective_cost_multiplier .* current_contributions) .+ public_share_each

            mean_rates[t] = mean(current_contributions ./ current_endowment)

            if t == rounds
                break
            end

            next_endowment = endowments[t + 1]
            next_grid = action_grid_for_endowment(next_endowment)
            next_contributions = Vector{Float64}(undef, GROUP_SIZE)

            volatility = t >= 2 ? abs(endowments[t] - endowments[t - 1]) / 100.0 : 0.0
            p_error = min(1.0, condition.base_error_prob + condition.error_sensitivity * volatility)

            for i in 1:GROUP_SIZE
                direction = if t == 1 || isnan(prev_subjective_payoff[i])
                    prev_direction[i]
                elseif subjective_payoff[i] > prev_subjective_payoff[i]
                    prev_direction[i]
                elseif subjective_payoff[i] < prev_subjective_payoff[i]
                    -prev_direction[i]
                else
                    prev_direction[i]
                end

                if condition.error_mode != "none" && rand(rng) < p_error
                    if condition.error_mode == "flip"
                        direction = direction == -1 ? 1 : -1
                    elseif condition.error_mode == "random"
                        direction = rand(rng, DIRECTIONS)
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
                next_contributions[i] = next_grid[new_index]

                prev_direction[i] = direction
            end

            prev_subjective_payoff = copy(subjective_payoff)
            current_contributions = next_contributions
        end

        return mean_rates
    end

    function run_task(task::TaskSpec)
        series = simulate_group(task.condition, task.rounds, task.seed)
        return task.condition_id, task.group_id, series
    end

    function build_tasks(conditions::Vector{Condition}, groups::Int, rounds::Int, base_seed::Int)
        tasks = TaskSpec[]
        sizehint!(tasks, length(conditions) * groups)
        for (condition_index, cond) in enumerate(conditions)
            cid = condition_id(cond)
            for group_id in 1:groups
                seed = base_seed + (condition_index - 1) * 1_000_000 + group_id
                push!(tasks, TaskSpec(cond, cid, group_id, rounds, seed))
            end
        end
        return tasks
    end

    function format_csv_value(value)
        if value isa Bool
            return value ? "True" : "False"
        elseif value isa AbstractFloat
            return string(value)
        else
            return string(value)
        end
    end

    function csv_escape(value)
        s = format_csv_value(value)
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

    function write_outputs(output_dir::AbstractString, conditions::Vector{Condition}, results_by_condition::Dict{String, Vector{Vector{Float64}}}, rounds::Int)
        condition_dir = joinpath(output_dir, "condition_csvs")
        mkpath(condition_dir)

        manifest_rows = ManifestRow[]
        combined_rows = CombinedRow[]

        round_headers = ["round_$(i)" for i in 1:rounds]

        for cond in conditions
            cid = condition_id(cond)
            ccode = condition_code(cond)
            filename = condition_filename(cond)
            csv_path = joinpath(condition_dir, filename)

            per_condition_rows = Vector{Vector{Any}}()
            condition_results = results_by_condition[cid]
            for (group_index, series) in enumerate(condition_results)
                row = Any[group_index]
                append!(row, series)
                push!(per_condition_rows, row)

                push!(combined_rows, CombinedRow(
                    cid,
                    ccode,
                    filename,
                    cond.condition_type,
                    cond.rho,
                    cond.sigma_e,
                    cond.base_error_prob,
                    cond.error_sensitivity,
                    cond.error_mode,
                    cond.earned,
                    cond.cost_weight_increase,
                    cond.gratitude_steps,
                    group_index,
                    copy(series),
                ))
            end

            write_csv(csv_path, vcat(["group_id"], round_headers), per_condition_rows)

            push!(manifest_rows, ManifestRow(
                cond.condition_type,
                cond.rho,
                cond.sigma_e,
                cond.base_error_prob,
                cond.error_sensitivity,
                cond.error_mode,
                cond.earned,
                cond.cost_weight_increase,
                cond.gratitude_steps,
                cid,
                ccode,
                filename,
                csv_path,
            ))
        end

        sort!(manifest_rows, by = row -> (row.condition_type, row.condition_code, row.condition_id))
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
            "csv_path",
        ]
        manifest_csv_rows = [
            Any[
                row.condition_type,
                row.rho,
                row.sigma_e,
                row.base_error_prob,
                row.error_sensitivity,
                row.error_mode,
                row.earned,
                row.cost_weight_increase,
                row.gratitude_steps,
                row.condition_id,
                row.condition_code,
                row.filename,
                row.csv_path,
            ]
            for row in manifest_rows
        ]
        write_csv(joinpath(output_dir, "manifest_conditions.csv"), manifest_header, manifest_csv_rows)

        sort!(combined_rows, by = row -> (row.condition_type, row.condition_code, row.group_id))
        combined_header = vcat(
            [
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
            ],
            round_headers,
        )
        combined_csv_rows = Vector{Vector{Any}}()
        for row in combined_rows
            csv_row = Any[
                row.condition_id,
                row.condition_code,
                row.filename,
                row.condition_type,
                row.rho,
                row.sigma_e,
                row.base_error_prob,
                row.error_sensitivity,
                row.error_mode,
                row.earned,
                row.cost_weight_increase,
                row.gratitude_steps,
                row.group_id,
            ]
            append!(csv_row, row.series)
            push!(combined_csv_rows, csv_row)
        end
        write_csv(joinpath(output_dir, "all_conditions_group_means.csv"), combined_header, combined_csv_rows)
    end
end

function run_all_conditions(; output_dir::AbstractString, groups::Int, rounds::Int, seed::Int, include_baseline::Bool, max_conditions::Union{Nothing, Int})
    mkpath(output_dir)

    conditions = build_conditions(include_baseline = include_baseline)
    if max_conditions !== nothing
        conditions = conditions[1:min(length(conditions), max_conditions)]
    end

    tasks = build_tasks(conditions, groups, rounds, seed)

    if isempty(tasks)
        error("No tasks were generated. Check condition settings.")
    end

    results_by_condition = Dict{String, Vector{Vector{Float64}}}()
    for cond in conditions
        results_by_condition[condition_id(cond)] = [Vector{Float64}(undef, rounds) for _ in 1:groups]
    end

    completed = Ref(0)
    total = length(tasks)
    progress_lock = ReentrantLock()

    function record_result!(cid::String, group_id::Int, series::Vector{Float64})
        lock(progress_lock) do
            results_by_condition[cid][group_id] = series
            completed[] += 1
            if completed[] % 1000 == 0 || completed[] == total
                println("Completed $(completed[])/$(total) simulations")
            end
        end
        return nothing
    end

    if isempty(Distributed.workers())
        for task in tasks
            cid, group_id, series = run_task(task)
            record_result!(cid, group_id, series)
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
                    cid, group_id, series = remotecall_fetch(run_task, pid, task)
                    record_result!(cid, group_id, series)
                end
            end
        end
    end

    write_outputs(output_dir, conditions, results_by_condition, rounds)

    println("Saved manifest to: $(joinpath(output_dir, "manifest_conditions.csv"))")
    println("Saved combined results to: $(joinpath(output_dir, "all_conditions_group_means.csv"))")
    println("Per-condition CSVs are in: $(joinpath(output_dir, "condition_csvs"))")
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
