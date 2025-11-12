using Distributed
using Random
using Dates
using Graphs
using SparseArrays
using LinearAlgebra

# -------- Parallel workers (bounded) --------
const _MAX_WORKERS = min(max(1, Sys.CPU_THREADS ÷ 2), 8)
if nprocs() == 1
    addprocs(_MAX_WORKERS; exeflags="--heap-size-hint=2G")
end

@everywhere begin
    using Random
    using Graphs
    using SparseArrays
    using LinearAlgebra
    try
        LinearAlgebra.BLAS.set_num_threads(1)
    catch
    end
end

# =========================
# Logging (master only)
# =========================
function save_simulation_log(script_path::String=abspath(PROGRAM_FILE), log_dir::String="logs")
    isdir(log_dir) || mkpath(log_dir)
    now = Dates.now()
    tsfmt = Dates.DateFormat("yyyymmdd_HHMMSS")
    isot  = Dates.DateFormat("yyyy-mm-dd\\THH:MM:SS")
    timestamp = Dates.format(now, tsfmt)

    script_log_path = joinpath(log_dir, "simulation_log_$(timestamp).txt")
    open(script_log_path, "w") do io
        write(io, "# Simulation started at: $(Dates.format(now, isot))\n\n")
        try
            if !isempty(script_path) && isfile(script_path)
                src = read(script_path, String)
                write(io, src)
            else
                write(io, "# [Warn] script source not available.\n")
            end
        catch
            write(io, "# [Warn] failed to read script source: $script_path\n")
        end
    end

    progress_log_path = joinpath(log_dir, "progress_log_$(timestamp).txt")
    println("ログファイル生成: $script_log_path")
    println("進捗ログファイル: $progress_log_path")
    return progress_log_path
end

@everywhere begin
    # ========== Constants ==========
    const num_agent = 400
    const num_periods = Int(400)
    const cost_of_cooperation = 1.0
    const num_generations = 1000
    const mutation_rate = 0.01
    const simulation = 10

    Random.seed!()  # per-process RNG

    # ========== Types ==========
    mutable struct Agent
        id::Int
        payoff::Float64
        norm::Vector{Char}           # length 4, 'G'/'B'
        reputation::Vector{Char}     # length num_agent, 'G'/'B'
    end

    mutable struct PublicInstitution
        norm::Vector{Char}           # length 4
        reputation::Vector{Char}     # length num_agent
    end

    # ========== Utils ==========
    normstring(v::Vector{Char}) = String(v)

    function new_agent(id::Int)
        norm = [rand(Bool) ? 'G' : 'B' for _ in 1:4]
        rep = fill('G', num_agent)
        rep[id] = 'G'
        return Agent(id, 0.0, norm, rep)
    end

    function decide_action(agent::Agent, recipient_rep::Char)::Char
        ns = normstring(agent.norm)
        if ns == "GGGG"
            return 'C'
        elseif ns == "BBBB"
            return 'D'
        end
        return recipient_rep == 'G' ? 'C' : 'D'
    end

    @inline function update_reputation_rule(norm::Vector{Char}, donor_action::Char, recipient_rep::Char)::Char
        if recipient_rep == 'G' && donor_action == 'C'
            return norm[1]
        elseif recipient_rep == 'G' && donor_action == 'D'
            return norm[2]
        elseif recipient_rep == 'B' && donor_action == 'C'
            return norm[3]
        else
            return norm[4]  # recipient_rep == 'B' && donor_action == 'D'
        end
    end

    function assign_roles(n::Int)
        perm = randperm(n)
        pairs = Vector{Tuple{Int,Int}}(undef, n)
        @inbounds for i in 1:n
            donor = perm[i]
            recipient = perm[i == n ? 1 : i+1]
            pairs[i] = (donor, recipient)
        end
        return pairs
    end

    function calculate_selection_weights(agents::Vector{Agent})
        payoffs = [a.payoff for a in agents]
        allsame = true
        first = payoffs[1]
        @inbounds for i in 2:length(payoffs)
            if payoffs[i] != first
                allsame = false
                break
            end
        end
        if allsame
            return fill(1.0/length(payoffs), length(payoffs))
        end
        minp = minimum(payoffs)
        shifted = [p - minp for p in payoffs]
        s = 0.0
        sq = similar(shifted)
        @inbounds for i in eachindex(shifted)
            v = shifted[i] * shifted[i]
            sq[i] = v
            s += v
        end
        if s == 0.0
            return fill(1.0/length(payoffs), length(payoffs))
        end
        return [x/s for x in sq]
    end

    @inline function sample_index(weights::Vector{Float64})::Int
        r = rand()
        acc = 0.0
        @inbounds for i in 1:length(weights)
            acc += weights[i]
            if r <= acc
                return i
            end
        end
        return length(weights)
    end

    # ========== Graph construction (Graphs.jl) ==========
    """
    build_observer_graph(N, network):
      0.0 -> empty graph (everyone public)
      1.0 -> nothing (well-mixed, no graph used)
      else -> regular ring with degree k = floor(network*N) (even, <= N-2)
    """
    function build_observer_graph(N::Int, network::Float64)
        if network == 0.0
            return SimpleGraph(N)        # empty graph
        elseif network == 1.0
            return nothing               # not used in well-mixed
        else
            k_raw = floor(Int, network * N)
            k = clamp(k_raw, 0, N - 2)
            if isodd(k); k = k > 0 ? k - 1 : 0; end
            if k == 0
                return SimpleGraph(N)
            end
            g = SimpleGraph(N)
            half = k ÷ 2
            @inbounds for i in 1:N
                # add edges only once (i < j) to avoid duplicates
                for s in 1:half
                    j = ((i - 1 + s) % N) + 1
                    if i < j
                        add_edge!(g, i, j)
                    end
                end
            end
            return g
        end
    end

    # ========== Warm-up (avoid simultaneous JIT spikes) ==========
    function _warmup_once()
        N = 8
        _ = assign_roles(N)
        _ = update_reputation_rule(['G','B','G','B'], 'C', 'G')
        g1 = build_observer_graph(N, 0.0)
        g2 = build_observer_graph(N, 0.5)
        g3 = build_observer_graph(N, 1.0)
        if g2 !== nothing
            _ = adjacency_matrix(g2)
        end
        a = Agent(1, 0.0, ['G','B','G','B'], fill('G', N))
        a.reputation[2] = 'G'
        _ = decide_action(a, a.reputation[2])
        return nothing
    end

    # ========== Main simulation of one setting ==========
    function run_simulation(params)
        error_rate_action, error_rate_evaluation, error_rate_public_evaluation,
        benefit_of_cooperation, network, public_norm_str, sim = params

        well_mixed  = (network == 1.0)  # private only
        public_only = (network == 0.0)  # public copy only

        # Graph only when needed
        g = build_observer_graph(num_agent, network)
        A = nothing
        if !(well_mixed || public_only)
            A = adjacency_matrix(g)  # SparseMatrixCSC{Int,Int}
        end

        # Initialization
        agents = [new_agent(i) for i in 1:num_agent]
        public_institution = PublicInstitution(collect(public_norm_str), fill('G', num_agent))

        cooperation_rates = Vector{Float64}()
        norm_distribution = Vector{Vector{Vector{Char}}}()

        for generation in 1:num_generations
            cooperation_count = 0
            interaction_count = 0

            for period in 1:num_periods
                pairs = assign_roles(num_agent)
                action_records = Vector{NTuple{3,Any}}()
                sizehint!(action_records, num_agent)

                for (donor_id, recipient_id) in pairs
                    donor = agents[donor_id]
                    recipient_rep = donor.reputation[recipient_id]

                    action = decide_action(donor, recipient_rep)
                    if rand() < error_rate_action
                        action = (action == 'C') ? 'D' : 'C'
                    end

                    push!(action_records, (donor_id, recipient_id, action))

                    # Public update (disabled in well-mixed)
                    if !well_mixed
                        pr = public_institution.reputation[recipient_id]
                        public_institution.reputation[donor_id] =
                            update_reputation_rule(public_institution.norm, action, pr)
                        if rand() < error_rate_public_evaluation
                            public_institution.reputation[donor_id] =
                                (public_institution.reputation[donor_id] == 'G') ? 'B' : 'G'
                        end
                    end

                    # Observers update: adjacent => private / non-adjacent => public
                    if well_mixed
                        @inbounds for evaluator_id in 1:num_agent
                            if evaluator_id == donor_id; continue; end
                            recipient_rep_e = agents[evaluator_id].reputation[recipient_id]
                            newrep = update_reputation_rule(agents[evaluator_id].norm, action, recipient_rep_e)
                            if rand() < error_rate_evaluation
                                newrep = (newrep == 'G') ? 'B' : 'G'
                            end
                            agents[evaluator_id].reputation[donor_id] = newrep
                        end
                    elseif public_only
                        repval = public_institution.reputation[donor_id]
                        @inbounds for evaluator_id in 1:num_agent
                            if evaluator_id == donor_id; continue; end
                            agents[evaluator_id].reputation[donor_id] = repval
                        end
                    else
                        @inbounds for evaluator_id in 1:num_agent
                            if evaluator_id == donor_id; continue; end
                            # adjacency check via sparse matrix
                            if A[donor_id, evaluator_id] != 0
                                recipient_rep_e = agents[evaluator_id].reputation[recipient_id]
                                newrep = update_reputation_rule(agents[evaluator_id].norm, action, recipient_rep_e)
                                if rand() < error_rate_evaluation
                                    newrep = (newrep == 'G') ? 'B' : 'G'
                                end
                                agents[evaluator_id].reputation[donor_id] = newrep
                            else
                                agents[evaluator_id].reputation[donor_id] = public_institution.reputation[donor_id]
                            end
                        end
                    end

                    # Cooperation rate tally (last 20% periods)
                    if period > Int(floor(0.8 * num_periods))
                        interaction_count += 1
                        if action == 'C'
                            cooperation_count += 1
                        end
                    end
                end

                # Apply payoffs in batch
                for rec in action_records
                    donor_id = rec[1]::Int
                    recipient_id = rec[2]::Int
                    action = rec[3]::Char
                    if action == 'C'
                        agents[donor_id].payoff -= cost_of_cooperation
                        agents[recipient_id].payoff += benefit_of_cooperation
                    end
                end
            end

            # Generation cooperation rate
            coop_rate = interaction_count == 0 ? 0.0 : cooperation_count / interaction_count
            push!(cooperation_rates, coop_rate)

            # Save norms for this generation
            push!(norm_distribution, [copy(a.norm) for a in agents])

            # Selection -> next gen
            weights = calculate_selection_weights(agents)
            new_agents = Vector{Agent}(undef, num_agent)
            for idx in 1:num_agent
                p1 = agents[sample_index(weights)]
                p2 = agents[sample_index(weights)]
                new_norm = Vector{Char}(undef, 4)
                @inbounds for gidx in 1:4
                    inherited = rand(Bool) ? p1.norm[gidx] : p2.norm[gidx]
                    new_norm[gidx] = (rand() < mutation_rate) ? ((inherited == 'G') ? 'B' : 'G') : inherited
                end
                rep = fill('G', num_agent); rep[idx] = 'G'
                new_agents[idx] = Agent(idx, 0.0, new_norm, rep)
            end
            agents = new_agents
            public_institution.reputation .= 'G'
        end

        # Filenames (same convention)
        file_key = "$(num_agent)_$(public_norm_str)_network$(network)_action_error$(error_rate_action)_evaluate_error$(error_rate_evaluation)_public_error$(error_rate_public_evaluation)_benefit$(benefit_of_cooperation)"
        file_prefix = file_key * "_$(sim+1)"

        # norm_distribution per Sim
        norm_file = "norm_distribution$(file_prefix).csv"
        open(norm_file, "w") do io
            write(io, "Generation")
            for i in 1:num_agent
                write(io, ",Agent_$(i)")
            end
            write(io, "\n")
            for (gen_idx, norms) in enumerate(norm_distribution)
                write(io, string(gen_idx-1))
                for norm in norms
                    write(io, ","); write(io, normstring(norm))
                end
                write(io, "\n")
            end
        end

        # return cooperation rates to master (aggregated output only)
        return Dict(
            "file_key" => file_key,
            "sim" => sim,
            "cooperation_rates" => cooperation_rates
        )
    end

    # Worker runner: no file/log writes here
    function run_and_collect(index_param)
        i, param = index_param
        result = run_simulation(param)
        return (i=i, result=result)
    end
end # @everywhere end

# ---- Sequential warm-up on each worker (avoid simultaneous JIT) ----
function _warm_all_workers_sequential()
    for pid in workers()
        remotecall_fetch(_warmup_once, pid)
        sleep(0.05)
    end
end

# =========================
# Main
# =========================
function main()
    progress_log_path = save_simulation_log()
    _warm_all_workers_sequential()

    error_rates_self = [0.001]
    error_rates_public = [0.001]
    benefit_values = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

    # 0.0=all public, 1.0=well-mixed (no public)
    network_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                      0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

    public_norms = ["GBBG", "GBGB", "GBGG", "GBBB"]

    params = Vector{NTuple{7,Any}}()
    for error_rate_action in error_rates_self
        error_rate_evaluation = error_rate_action
        for error_rate_public_evaluation in error_rates_public
            for benefit_of_cooperation in benefit_values
                for network in network_values
                    for public_norm in public_norms
                        for sim in 0:(simulation-1)
                            push!(params, (error_rate_action, error_rate_evaluation, error_rate_public_evaluation,
                                           benefit_of_cooperation, network, public_norm, sim))
                        end
                    end
                end
            end
        end
    end

    idx_params = collect(enumerate(params))

    # Run in parallel with gentle batching
    wrapped = pmap(idx_params; batch_size=1) do ip
        run_and_collect(ip)
    end

    # progress log (master only)
    open(progress_log_path, "a") do io
        for w in wrapped
            fk = w.result["file_key"]; sm = w.result["sim"]
            write(io, "[ $(w.i)/$(length(idx_params)) ] Finished: $(fk) (Sim $(sm))\n")
        end
    end

    # Aggregate cooperation rates per setting (one CSV per setting)
    grouped = Dict{String, Dict{String, Vector{Float64}}}()  # key => "SimX" => rates
    max_gen = 0
    for w in wrapped
        r = w.result
        key = r["file_key"]::String
        sim = r["sim"]
        rates = r["cooperation_rates"]::Vector{Float64}
        haskey(grouped, key) || (grouped[key] = Dict{String, Vector{Float64}}())
        grouped[key]["Sim$(sim)"] = rates
        max_gen = max(max_gen, length(rates))
    end

    for (key, simdict) in grouped
        safe_key = replace(key, '.' => 'p')
        out = "cooperation_rates_$(safe_key).csv"
        try
            open(out, "w") do io
                write(io, "Generation")
                simnames = sort(collect(keys(simdict)))  # "Sim0","Sim1",...
                for s in simnames
                    write(io, ","); write(io, s)
                end
                write(io, "\n")
                for gen in 1:max_gen
                    write(io, string(gen))
                    for s in simnames
                        rates = simdict[s]
                        val = gen <= length(rates) ? rates[gen] : ""
                        write(io, ","); write(io, string(val))
                    end
                    write(io, "\n")
                end
            end
        catch e
            @warn "Failed to write aggregated csv" file=out exception=(e, catch_backtrace())
        end
    end

    # Graceful shutdown (avoid concurrent teardown)
    try
        GC.gc()
        for pid in workers()
            rmprocs(pid; waitfor=5)
            sleep(0.1)
        end
        GC.gc()
    catch e
        @warn "graceful shutdown failed" exception=(e, catch_backtrace())
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
