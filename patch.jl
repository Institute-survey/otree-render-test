# ==== results から協力率CSVを出力するワンショットスクリプト ====

# 1) results があるか確認
if !isdefined(Main, :results)
    error("このセッションに `results` が見つかりません。先にシミュレーションで `results` を生成した同一セッションで実行してください。")
end

# 2) 型に依存しないよう Any 扱いで取り出すヘルパ
getkey(d, k) = haskey(d, k) ? d[k] : nothing

# 3) 結果の集約（メモリ上の results から構築）
grouped = Dict{String, Dict{String, Vector{Float64}}}()  # key => ("SimX" => rates)
max_gen = 0
bad = 0
for (i, r) in enumerate(results)
    try
        key  = String(getkey(r, "file_key"))
        sim  = getkey(r, "sim")
        rates = Vector{Float64}(getkey(r, "cooperation_rates"))
        haskey(grouped, key) || (grouped[key] = Dict{String, Vector{Float64}}())
        grouped[key]["Sim$(sim)"] = rates
        max_gen = max(max_gen, length(rates))
    catch e
        @warn "results[$i] の読み取りに失敗しました。スキップします。" exception=(e, catch_backtrace())
        bad += 1
    end
end

isempty(grouped) && error("有効な協力率データが見つかりませんでした。")

# 4) 出力先ディレクトリを用意
outdir = "out_coop"
isdir(outdir) || mkpath(outdir)

# 5) 各Simごとの縦持ちCSV（Generation,CooperationRate）
for (key, simdict) in grouped
    safe_key = replace(key, '.' => 'p')   # ファイル名の安全化
    for (simname, rates) in simdict
        out_sim = joinpath(outdir, "cooperation_rates_${safe_key}_${simname}.csv")
        open(out_sim, "w") do io
            write(io, "Generation,CooperationRate\n")
            for (gen, rate) in enumerate(rates)
                write(io, string(gen))   # 1始まり（0始まりにしたい場合は gen-1）
                write(io, ","); write(io, string(rate)); write(io, "\n")
            end
        end
        @info "Wrote per-sim CSV" file=out_sim
    end
end

# 6) 条件ごとの横持ちCSV（Sim列が横に並ぶ）
for (key, simdict) in grouped
    safe_key = replace(key, '.' => 'p')
    out = joinpath(outdir, "cooperation_rates_${safe_key}.csv")
    open(out, "w") do io
        # ヘッダ
        write(io, "Generation")
        simnames = sort(collect(keys(simdict)))  # "Sim0","Sim1",...
        for s in simnames
            write(io, ","); write(io, s)
        end
        write(io, "\n")
        # 行
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
    @info "Wrote aggregated CSV" file=out
end

@info "Done. CSV files are under $(abspath(outdir))." skipped_items=bad
