using Distributed
using Random
using Dates

# --- 並列ワーカーの用意 ---
if nprocs() == 1
    addprocs()
end

# =========================
# ログ関数（マスターのみ）
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
                write(io, "# [Warn] script source not available in this context.\n")
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
    using Random

    # ========== 初期値 ==========
    const num_agent = 400              # エージェント数
    const num_periods = Int(400)       # ピリオド数
    const num_generations = 1000       # ジェネレーション数
    const mutation_rate = 0.01         # 突然変異確率
    const simulation = 10              # シミュレーション回数

    # 乱数は各プロセスで独立に初期化（シード固定したい場合は適宜設定）
    Random.seed!()

    # ========== 型定義 ==========
    mutable struct Agent
        id::Int
        payoff::Float64
        norm::Vector{Char}           # 長さ4、'G'/'B'
        reputation::Vector{Char}     # 長さ num_agent、'G'/'B'
    end

    mutable struct PublicInstitution
        norm::Vector{Char}           # 長さ4
        reputation::Vector{Char}     # 長さ num_agent
    end

    # ========== ユーティリティ ==========
    normstring(v::Vector{Char}) = String(v)

    function random_reputation()
        return [rand(Bool) ? 'G' : 'B' for _ in 1:num_agent]
    end

    function new_agent(id::Int)
        # 規範は G/B を長さ4でランダム
        norm = [rand(Bool) ? 'G' : 'B' for _ in 1:4]
        # 評判ベクトルも全員分 G/B ランダム
        rep = random_reputation()
        Agent(id, 0.0, norm, rep)
    end

    function decide_action(agent::Agent, recipient_rep::Char)::Char
        return recipient_rep == 'G' ? 'C' : 'D'
    end

    @inline function update_reputation_rule(norm::Vector{Char}, donor_action::Char, recipient_rep::Char)::Char
        if recipient_rep == 'G' && donor_action == 'C'
            return norm[1]
        elseif recipient_rep == 'G' && donor_action == 'D'
            return norm[2]
        elseif recipient_rep == 'B' && donor_action == 'C'
            return norm[3]
        else # recipient_rep == 'B' && donor_action == 'D'
            return norm[4]
        end
    end

    # ====== 変動コスト・ベネフィット（今回の修正点） ======
    # 協力（C）の時のドナーにかかるコスト：0〜+5（整数一様）
    # 非協力（D）の時のドナーにかかるコスト：-5~0（整数一様）
    # 協力（C）の時のレシピエントにわたるベネフィット：1〜+5（整数一様）
    # 非協力（D）の時のレシピエントにわたるベネフィット：-5〜0（整数一様）
    @inline function draw_donor_cost(action::Char)::Float64
        if action == 'C'
            return Float64(rand(-0:5))
        else
            return Float64(rand(-5:0))
        end
    end

    @inline function draw_recipient_benefit(action::Char)::Float64
        if action == 'C'
            return Float64(rand(1:5))
        else
            return Float64(rand(-5:0))
        end
    end

    # 役割割り当て：ランダム並べ替え→(i, i+1 mod N)
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

    # 選択重み（@simd/break併用は避ける）
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
            return fill(1.0 / length(payoffs), length(payoffs))
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
            return fill(1.0 / length(payoffs), length(payoffs))
        end
        return [x / s for x in sq]
    end

    # 重み付きサンプリング（1個を復元抽出）
    @inline function sample_index(weights::Vector{Float64})::Int
        r = rand()
        acc = 0.0
        @inbounds for i in 1:length(weights)
            acc += weights[i]
            if r <= acc
                return i
            end
        end
        return length(weights) # 端数のため
    end

    # 1条件のシミュレーション本体
    function run_simulation(params)
        # benefit_of_cooperation は廃止（変動にしたため）
        error_rate_action, error_rate_evaluation, error_rate_public_evaluation, probability, public_norm_str, sim = params

        # 初期化
        agents = [new_agent(i) for i in 1:num_agent]
        public_institution = PublicInstitution(
            collect(public_norm_str),
            random_reputation()
        )

        cooperation_rates = Vector{Float64}()
        norm_distribution = Vector{Vector{Vector{Char}}}()  # genごとの全agentのnorm

        for generation in 1:num_generations
            cooperation_count = 0
            interaction_count = 0

            for period in 1:num_periods
                pairs = assign_roles(num_agent)

                # (donor, recipient, action, donor_cost, recipient_benefit)
                action_records = Vector{Tuple{Int,Int,Char,Float64,Float64}}()
                sizehint!(action_records, num_agent)

                for (donor_id, recipient_id) in pairs
                    donor = agents[donor_id]
                    recipient_rep = donor.reputation[recipient_id]

                    action = decide_action(donor, recipient_rep)
                    if rand() < error_rate_action
                        action = (action == 'C') ? 'D' : 'C'
                    end

                    # 今回の修正：行動ごとにコスト・ベネフィットを乱数で決めて記録
                    dcost = draw_donor_cost(action)
                    rben  = draw_recipient_benefit(action)
                    push!(action_records, (donor_id, recipient_id, action, dcost, rben))

                    # --- 評判更新（即時） ---
                    # 公的評価
                    pr = public_institution.reputation[recipient_id]
                    public_institution.reputation[donor_id] =
                        update_reputation_rule(public_institution.norm, action, pr)
                    if rand() < error_rate_public_evaluation
                        public_institution.reputation[donor_id] =
                            (public_institution.reputation[donor_id] == 'G') ? 'B' : 'G'
                    end

                    # 評価者集合の生成
                    for evaluator_id in 1:num_agent
                        if evaluator_id == donor_id
                            continue
                        end
                        if rand() > probability
                            # 私的評価
                            recipient_rep_e = agents[evaluator_id].reputation[recipient_id]
                            newrep = update_reputation_rule(agents[evaluator_id].norm, action, recipient_rep_e)
                            if rand() < error_rate_evaluation
                                newrep = (newrep == 'G') ? 'B' : 'G'
                            end
                            agents[evaluator_id].reputation[donor_id] = newrep
                        else
                            # 公的評価に従う
                            agents[evaluator_id].reputation[donor_id] = public_institution.reputation[donor_id]
                        end
                    end

                    # 協力率の集計（最後の20%期間）
                    if period > Int(floor(0.8 * num_periods))
                        interaction_count += 1
                        if action == 'C'
                            cooperation_count += 1
                        end
                    end
                end

                # === 利得の一括適用（今回の修正：C/Dどちらでも適用） ===
                @inbounds for rec in action_records
                    donor_id, recipient_id, _action, dcost, rben = rec
                    # ドナーは「コスト」を差し引く（dcost が負なら増える）
                    agents[donor_id].payoff -= dcost
                    # レシピエントは「ベネフィット」を加算（rben が負なら減る）
                    agents[recipient_id].payoff += rben
                end
            end

            # 世代の協力率
            coop_rate = interaction_count == 0 ? 0.0 : cooperation_count / interaction_count
            push!(cooperation_rates, coop_rate)

            # 規範分布の保存（各エージェントのnorm）
            push!(norm_distribution, [copy(a.norm) for a in agents])

            # 次世代へ（選択）
            weights = calculate_selection_weights(agents)
            new_agents = Vector{Agent}(undef, num_agent)
            for idx in 1:num_agent
                p1 = agents[sample_index(weights)]
                p2 = agents[sample_index(weights)]
                new_norm = Vector{Char}(undef, 4)
                @inbounds for g in 1:4
                    inherited = rand(Bool) ? p1.norm[g] : p2.norm[g]
                    if rand() < mutation_rate
                        new_norm[g] = (inherited == 'G') ? 'B' : 'G'
                    else
                        new_norm[g] = inherited
                    end
                end
                # 次世代エージェントの初期評判も G/B ランダム
                rep = random_reputation()
                new_agents[idx] = Agent(idx, 0.0, new_norm, rep)
            end
            agents = new_agents

            # 公的機関の評判も世代ごとに G/B ランダムにリセット
            public_institution.reputation = random_reputation()
        end

        # ファイル名キーの作成（benefit は廃止）
        file_prefix_without_sim = "$(num_agent)_$(public_norm_str)_probability$(probability)_action_error$(error_rate_action)_evaluate_error$(error_rate_evaluation)_public_error$(error_rate_public_evaluation)"
        file_prefix = file_prefix_without_sim * "_$(sim+1)"

        # 規範分布CSVの保存
        norm_file = "norm_distribution$(file_prefix).csv"
        open(norm_file, "w") do io
            # ヘッダ
            write(io, "Generation")
            for i in 1:num_agent
                write(io, ",Agent_$(i)")
            end
            write(io, "\n")
            # 各世代（Pythonに合わせて0始まりで出力）
            for (gen_idx, norms) in enumerate(norm_distribution)
                write(io, string(gen_idx - 1))
                for norm in norms
                    write(io, ",")
                    write(io, normstring(norm))
                end
                write(io, "\n")
            end
        end

        return Dict(
            "file_key" => file_prefix_without_sim,
            "sim" => sim,
            "cooperation_rates" => cooperation_rates
        )
    end

    # 進捗ログ付き実行
    function run_and_log(index_param, progress_log_path::String, total::Int)
        i, param = index_param
        result = run_simulation(param)
        fk = result["file_key"]
        sm = result["sim"]
        open(progress_log_path, "a") do io
            write(io, "[ $(i+1)/$total ] Finished: $(fk) (Sim $(sm))\n")
        end
        return result
    end
end # @everywhere

# --- メイン ---
function main()
    # ログはマスターでのみ作成
    progress_log_path = save_simulation_log()

    error_rates_self = [0.001, 0.01, 0.05]
    error_rates_public = [0.001]
    probability_values = [0.0]
    public_norms = ["GBBG"]

    # params: (error_rate_action, error_rate_evaluation, error_rate_public_evaluation, probability, public_norm, sim)
    params = Vector{NTuple{6,Any}}()
    for error_rate_action in error_rates_self
        error_rate_evaluation = error_rate_action
        for error_rate_public_evaluation in error_rates_public
            for probability in probability_values
                for public_norm in public_norms
                    for sim in 0:(simulation-1)
                        push!(params, (error_rate_action, error_rate_evaluation, error_rate_public_evaluation,
                                       probability, public_norm, sim))
                    end
                end
            end
        end
    end

    total = length(params)
    idx_params = collect(enumerate(params))

    const_progress = progress_log_path
    const_total = total

    results = pmap(idx_params) do ip
        run_and_log(ip, const_progress, const_total)
    end

    # --- 結果の集約 ---
    grouped = Dict{String, Dict{String, Vector{Float64}}}()
    max_gen = 0
    for result in results
        key = result["file_key"]::String
        sim = result["sim"]
        coop_rates = result["cooperation_rates"]::Vector{Float64}
        haskey(grouped, key) || (grouped[key] = Dict{String, Vector{Float64}}())
        grouped[key]["Sim$(sim)"] = coop_rates
        max_gen = max(max_gen, length(coop_rates))
    end

    # 条件ごとにCSV出力（列=各Sim）
    for (key, simdict) in grouped
        out = "cooperation_rates_$(key).csv"
        open(out, "w") do io
            # ヘッダ
            write(io, "Generation")
            simnames = sort(collect(keys(simdict)))  # "Sim0","Sim1",...
            for s in simnames
                write(io, ",")
                write(io, s)
            end
            write(io, "\n")
            # 行
            for gen in 1:max_gen
                write(io, string(gen))
                for s in simnames
                    rates = simdict[s]
                    val = gen <= length(rates) ? rates[gen] : ""
                    write(io, ",")
                    write(io, string(val))
                end
                write(io, "\n")
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
