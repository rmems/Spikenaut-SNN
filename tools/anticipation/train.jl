#!/usr/bin/env julia
# Start before imports so compilation/loading belongs to the shared runtime budget.
const ANTICIPATION_STARTED = time_ns() / 1e9
module AnticipationTrainer
using JSON3, Random, Statistics, LinearAlgebra, SHA
using SynapticDistill
export new_model, target_normalization, run_rows!, save_checkpoint, load_checkpoint, prepare_rows, campaign

const DT = 0.1f0
const TRACE_TAU = 0.5f0
const TRACE_LAMBDA = exp(-DT / TRACE_TAU)
const LEARNING_RATE = 0.01f0
const BATCH_SIZE = 64

mutable struct ForecastSNN
    arm::String
    seed::Int
    input_weights::Matrix{Float32}
    weights::Matrix{Float32}
    tau::Vector{Float32}
end

function new_model(arm, seed)
    arm in ("uniform", "mixed") || error("Unknown memory arm: $arm")
    rng = MersenneTwister(seed)
    input = (0.6f0 / sqrt(5f0)) .* randn(rng, Float32, 16, 5)
    weights = 0.02f0 .* randn(rng, Float32, 4, 16)
    tau = arm == "uniform" ? fill(0.5f0, 16) : repeat(Float32[0.1, 0.5, 2, 5], inner=4)
    ForecastSNN(arm, seed, input, weights, tau)
end

function target_normalization(rows)
    ys = [Float64.(r["y"]) for r in rows if r["split"] == "train" && r["eligible"]]
    isempty(ys) && error("No eligible training targets")
    y = hcat(ys...)
    mu = vec(mean(y; dims=2))
    raw = vec(std(y; dims=2, corrected=false))
    Dict("y_mean"=>mu, "y_std"=>[s > 0 ? s : 1.0 for s in raw])
end

function prepare_rows(prepared)
    prepared["schema_version"] == "anticipation-prepared-v1" || error("Unsupported prepared schema")
    norm = prepared["normalization"]
    norm["fit_split"] == "train" || error("Normalization must be fitted on training only")
    xm, xs = Float32.(norm["x_mean"]), Float32.(norm["x_std"])
    length(xm) == length(xs) == 5 || error("Expected five input features")
    all(isfinite, xm) && all(x->isfinite(x) && x > 0, xs) || error("Invalid normalization")
    rows = Any[]
    for session in prepared["sessions"]
        split = String(session["split"])
        split in ("train", "validation", "test") || error("Unknown split $split")
        examples = Dict(Int(e["frame_index"])=>e for e in session["examples"])
        for (i, frame) in enumerate(session["frames"])
            valid = Bool(frame["valid"])
            eligible = haskey(examples, i-1)
            eligible && !valid && error("Eligible invalid frame")
            x = valid ? (Float32.(frame["x"]) .- xm) ./ xs : zeros(Float32, 5)
            all(isfinite, x) || error("Nonfinite feature")
            y = eligible ? Float32.(examples[i-1]["y"]) : zeros(Float32, 4)
            length(y) == 4 && all(isfinite, y) || error("Invalid target")
            push!(rows, Dict("session_id"=>String(session["session_id"]), "split"=>split,
                "segment_id"=>frame["segment_id"], "frame_index"=>i-1,
                "valid"=>valid, "eligible"=>eligible, "x"=>x, "y"=>y))
        end
    end
    for split in ("train", "validation", "test")
        any(r->r["split"]==split && r["eligible"], rows) || error("No eligible examples in $split")
    end
    rows, Dict("y_mean"=>Float64.(norm["y_mean"]), "y_std"=>Float64.(norm["y_std"]))
end

struct BudgetExpired <: Exception end
monotonic_seconds() = time_ns() / 1e9
check_budget(deadline) = monotonic_seconds() >= deadline ? throw(BudgetExpired()) : nothing

"""Process every causal valid tick, including warmup. State is local and reset at gaps.
Readout uses the same normalized exponential spike trace as SynapticDistill OTTT.
Loss is a sum over time divided by outputs; OTTT itself averages time. Mask
compensation T/n ensures warmup never dilutes the mean eligible-example gradient.
"""
function run_rows!(model, rows, norm; train=false, deadline=Inf)
    predictions = Any[]
    spikes_total = zeros(Int, 16)
    tick_count = 0
    max_columns = 0
    squared_error = 0.0
    supervised_count = 0
    time_resolved = false
    membrane = zeros(Float32, 16)
    pre = zeros(Float32, 16)
    decay = exp.(-DT ./ model.tau)
    ym, ys = Float32.(norm["y_mean"]), Float32.(norm["y_std"])
    all(x->isfinite(x) && x > 0, ys) || error("Invalid target scale")
    previous = nothing
    previous_index = -2
    trace = nothing
    # Segment boundaries are explicit; invalid frames additionally break continuity.
    groups = Vector{Vector{Any}}()
    for row in rows
        if !get(row, "valid", true)
            previous = nothing
            previous_index = -2
            continue
        end
        key = (row["session_id"], row["segment_id"])
        if key != previous || row["frame_index"] != previous_index + 1
            push!(groups, Any[])
        end
        push!(groups[end], row)
        previous, previous_index = key, row["frame_index"]
    end
    for group in groups
        fill!(membrane, 0)
        fill!(pre, 0)
        trace = nothing
        for start in 1:BATCH_SIZE:length(group)
            check_budget(deadline)
            batch = group[start:min(start+BATCH_SIZE-1, length(group))]
            # SynapticDistill refuses an ambiguous 16×16 spike layout. Divide such
            # tails into 15+1 without introducing a state reset or dummy tick.
            chunks = length(batch)==16 ? (batch[1:15], batch[16:16]) : (batch,)
            for chunk in chunks
                T = length(chunk)
                S, F, Y = zeros(Float32, 16, T), zeros(Float32, 16, T), zeros(Float32, 4, T)
                mask = zeros(Float32, 1, T)
                for (t, row) in enumerate(chunk)
                    current = 1.2f0 .+ model.input_weights * Float32.(row["x"])
                    membrane .= decay .* membrane .+ (1f0 .- decay) .* current
                    s = Float32.(membrane .>= 1f0)
                    membrane .-= s
                    pre .= TRACE_LAMBDA .* pre .+ s
                    S[:,t] .= s
                    F[:,t] .= (1f0-TRACE_LAMBDA) .* pre
                    spikes_total .+= Int.(s)
                    tick_count += 1
                    if row["eligible"]
                        Y[:,t] .= (Float32.(row["y"]) .- ym) ./ ys
                        mask[t] = 1f0
                    end
                end
                output = (logits=model.weights * F,)
                for (t, row) in enumerate(chunk)
                    row["eligible"] || continue
                    push!(predictions, Dict("session_id"=>row["session_id"],
                        "frame_index"=>row["frame_index"], "split"=>row["split"],
                        "prediction"=>Float64.(output.logits[:,t] .* ys .+ ym)))
                end
                n = sum(mask)
                squared_error += sum(abs2, (output.logits .- Y) .* mask)
                supervised_count += Int(n)
                if train && n > 0
                    loss_fn = o -> sum(abs2, (o.logits .- Y) .* mask) * (Float32(T) / (4f0*n))
                    grads, trace = SynapticDistill.update_ottt!(model,
                        SynapticDistill.SpikeBatch(S, nothing, nothing), loss_fn(output), output;
                        traces=trace, trace_lambda=TRACE_LAMBDA, loss_fn=loss_fn)
                    trace.traces.time_resolved || error("OTTT lost time resolution")
                    time_resolved = true
                    max_columns = max(max_columns, T)
                    model.weights .-= LEARNING_RATE .* grads
                    all(isfinite, model.weights) || error("Nonfinite readout weights")
                elseif train
                    # Warmup must seed the subsequent OTTT eligibility trace.
                    trace = SynapticDistill.TraceBatch((pre=copy(pre),))
                end
            end
        end
    end
    diagnostics = Dict("ticks"=>tick_count, "spikes_per_neuron"=>spikes_total,
        "silent_neurons"=>findall(==(0), spikes_total), "spike_rate_hz"=>spikes_total ./ max(tick_count*Float64(DT), eps()),
        "time_resolved_ottt"=>time_resolved, "max_time_columns"=>max_columns)
    loss = supervised_count > 0 ? squared_error / (4supervised_count) : nothing
    (;predictions, diagnostics, time_resolved, max_time_columns=max_columns, loss)
end

function primary_score(predictions, rows, norm)
    truths = Dict((r["session_id"],r["frame_index"])=>r["y"] for r in rows if r["eligible"])
    sessions = Dict{String,Vector{Float64}}()
    for p in predictions
        y = truths[(p["session_id"],p["frame_index"])]
        e = mean(abs((p["prediction"][j]-y[j])/norm["y_std"][j]) for j in (3,4))
        push!(get!(sessions, p["session_id"], Float64[]), e)
    end
    isempty(sessions) && error("Cannot score empty validation predictions")
    mean(mean(errors) for errors in values(sessions))
end

matrix_rows(m) = [collect(m[i,:]) for i in axes(m,1)]
function write_json(path, data)
    mkpath(dirname(path))
    temp, io = mktemp(dirname(path); cleanup=false)
    try
        JSON3.write(io, data)
        close(io)
        mv(temp, path; force=true)
    finally
        close(io)
        rm(temp; force=true)
    end
end
function save_checkpoint(path, model, norm; metadata=Dict())
    write_json(path, Dict("schema_version"=>"anticipation-snn-checkpoint-v1",
        "arm"=>model.arm,"seed"=>model.seed,"input_weights"=>matrix_rows(model.input_weights),
        "readout_weights"=>matrix_rows(model.weights),"membrane_tau_seconds"=>model.tau,
        "dt_seconds"=>DT,"readout_trace_tau_seconds"=>TRACE_TAU,"hidden_bias"=>1.2,
        "threshold"=>1.0,"reset"=>"subtract_threshold","normalization"=>norm,
        "metadata"=>metadata))
end
function load_checkpoint(path)
    d = JSON3.read(read(path,String), Dict{String,Any})
    d["schema_version"] == "anticipation-snn-checkpoint-v1" || error("Unsupported checkpoint")
    model = ForecastSNN(d["arm"], d["seed"], reduce(vcat,permutedims.(Float32.(r) for r in d["input_weights"])),
        reduce(vcat,permutedims.(Float32.(r) for r in d["readout_weights"])), Float32.(d["membrane_tau_seconds"]))
    model, d["normalization"]
end

function campaign(prepared_path, output, budget_seconds; started=monotonic_seconds(), epochs=20)
    0 <= epochs <= 20 || error("Epoch count must be 0..20")
    budget_seconds >= 0 || error("Budget must be nonnegative")
    deadline = started + budget_seconds
    # Reserve 15% of this same budget for held-out predictions and artifact writes.
    training_deadline = started + 0.85budget_seconds
    prepared = JSON3.read(read(prepared_path,String), Dict{String,Any})
    rows, norm = prepare_rows(prepared)
    train = filter(r->r["split"]=="train", rows)
    validation = filter(r->r["split"]=="validation", rows)
    heldout = filter(r->r["split"] in ("validation","test"), rows)
    provenance = Dict("julia_version"=>string(VERSION),
        "synaptic_distill_entrypoint_sha256"=>bytes2hex(sha256(read(pathof(SynapticDistill)))),
        "synaptic_distill_ottt_sha256"=>bytes2hex(sha256(read(joinpath(dirname(pathof(SynapticDistill)),"rules","ottt.jl")))),
        "synaptic_distill_utils_sha256"=>bytes2hex(sha256(read(joinpath(dirname(pathof(SynapticDistill)),"utils.jl")))),
        "prepared_sha256"=>bytes2hex(sha256(read(prepared_path))),
        "source_provenance"=>prepared["provenance"], "feature_map_id"=>prepared["feature_map_id"],
        "feature_map"=>prepared["feature_map"],
        "target_names"=>prepared["target_names"], "input_normalization"=>prepared["normalization"],
        "selection"=>"validation equal-session mean 5s standardized MAE", "learning_rate"=>LEARNING_RATE)
    models = [new_model(arm,seed) for seed in (123,456,789) for arm in ("uniform","mixed")]
    results = [Dict{String,Any}("arm"=>m.arm,"seed"=>m.seed,"status"=>"unfinished",
        "epochs_completed"=>0,"selected_epoch"=>nothing,"validation_primary"=>nothing,
        "predictions"=>nothing,"checkpoint"=>nothing) for m in models]
    curves = [Any[] for _ in models]
    # Round robin prevents the first arm from consuming the shared budget.
    for epoch in 1:epochs
        monotonic_seconds() >= training_deadline && break
        for (idx,m) in enumerate(models)
            monotonic_seconds() >= training_deadline && break
            try
                tr = run_rows!(m,train,norm;train=true,deadline=training_deadline)
                val = run_rows!(m,validation,norm;deadline=training_deadline)
                score = primary_score(val.predictions,validation,norm)
                push!(curves[idx],Dict("epoch"=>epoch,"validation_primary"=>score,
                    "train_loss"=>tr.loss, "validation_loss"=>val.loss,
                    "training_diagnostics"=>tr.diagnostics, "elapsed_seconds"=>monotonic_seconds()-started))
                r = results[idx]
                r["epochs_completed"] = epoch
                println("arm=$(m.arm) seed=$(m.seed) epoch=$epoch train_mse=$(tr.loss) validation_mse=$(val.loss) validation_primary=$score elapsed_seconds=$(monotonic_seconds()-started)")
                flush(stdout)
                if r["validation_primary"] === nothing || score < r["validation_primary"]
                    r["validation_primary"], r["selected_epoch"] = score,epoch
                    relative = "$(m.arm)-$(m.seed)/checkpoint.json"
                    save_checkpoint(joinpath(output,relative),m,norm;metadata=merge(provenance,Dict("epoch"=>epoch)))
                    r["checkpoint"] = relative
                end
            catch err
                err isa BudgetExpired || rethrow()
                break
            end
        end
    end
    for (idx,r) in enumerate(results)
        subdir = "$(r["arm"])-$(r["seed"])"
        write_json(joinpath(output,subdir,"learning_curves.json"),curves[idx])
        r["stop_reason"] = r["epochs_completed"]==epochs ? "epoch_limit" : "shared_budget"
        r["checkpoint"] === nothing && continue
        try
            m, restored_norm = load_checkpoint(joinpath(output,r["checkpoint"]))
            ev = run_rows!(m,heldout,restored_norm;deadline=deadline)
            relative = joinpath(subdir,"predictions.json")
            write_json(joinpath(output,relative),Dict("schema_version"=>"anticipation-predictions-v1",
                "arm"=>m.arm,"seed"=>m.seed,"predictions"=>ev.predictions,"diagnostics"=>ev.diagnostics))
            r["predictions"] = relative
            r["status"] = r["epochs_completed"]==epochs ? "complete" : "budget_limited"
        catch err
            err isa BudgetExpired || rethrow()
            r["status"] = "unfinished_evaluation"
        end
    end
    summary = Dict("schema_version"=>"anticipation-snn-runs-v1","runs"=>results,
        "budget_seconds"=>budget_seconds,"elapsed_seconds"=>monotonic_seconds()-started,
        "training_fraction"=>0.85,"max_epochs"=>epochs,"prepared_sha256"=>provenance["prepared_sha256"])
    write_json(joinpath(output,"summary.json"),summary)
    summary
end
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS)==3 || error("Usage: train.jl PREPARED_JSON OUTPUT_DIRECTORY BUDGET_SECONDS")
    AnticipationTrainer.campaign(ARGS[1],ARGS[2],parse(Float64,ARGS[3]);started=ANTICIPATION_STARTED)
end
