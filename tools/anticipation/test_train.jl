using Test
include(joinpath(@__DIR__, "train.jl"))
using .AnticipationTrainer

function fixture()
    rows = Any[]
    for (sid, split) in [("s1", "train"), ("s2", "validation"), ("s3", "test")]
        for i in 0:139
            x = Float32[sin(i / 9), cos(i / 13), sin(i / 17), cos(i / 7), 0.5]
            push!(rows, Dict("session_id"=>sid, "split"=>split, "segment_id"=>0,
                "frame_index"=>i, "x"=>x, "eligible"=>i>=50,
                "y"=>Float32[sin(i/9), cos(i/13), sin(i/17), cos(i/7)]))
        end
    end
    rows
end

@testset "Time-resolved learning, evaluation, reset and checkpoint" begin
    rows = fixture()
    m = new_model("uniform", 123)
    mixed = new_model("mixed", 123)
    @test m.input_weights == mixed.input_weights
    @test m.weights == mixed.weights
    norm = target_normalization(rows)
    @test norm == target_normalization(filter(r->r["split"]=="train", rows))
    before = copy(m.weights)
    train = filter(r->r["split"]=="train", rows)
    r = run_rows!(m, train, norm; train=true)
    @test m.weights != before
    @test r.time_resolved
    @test r.max_time_columns > 1
    learned = copy(m.weights)
    ev = run_rows!(m, rows, norm)
    @test m.weights == learned
    @test length(ev.predictions) == 270
    @test length(unique(p["prediction"] for p in ev.predictions)) > 1
    alone = run_rows!(m, filter(r->r["split"]=="test", rows), norm)
    @test alone.predictions == filter(p->p["split"]=="test", ev.predictions)
    changed_targets = deepcopy(rows)
    for row in changed_targets
        row["split"] == "train" && continue
        row["y"] .= 999
    end
    @test run_rows!(m, changed_targets, norm).predictions == ev.predictions
    @test target_normalization(changed_targets) == norm
    # Splitting a previously driven stream must reset both membrane and readout trace.
    segment = deepcopy(train[51:100])
    for r in segment
        r["segment_id"] = 1
    end
    standalone = run_rows!(m, segment, norm)
    joined = run_rows!(m, vcat(train[1:50], segment), norm)
    @test joined.predictions == standalone.predictions
    gap = deepcopy(train[51])
    gap["valid"] = false
    after_gap = deepcopy(train[52:100])
    @test run_rows!(m, vcat(train[1:50], [gap], after_gap), norm).predictions ==
          run_rows!(m, after_gap, norm).predictions
    mktempdir() do path
        file = joinpath(path, "checkpoint.json")
        save_checkpoint(file, m, norm; metadata=Dict("test"=>true))
        restored, restored_norm = load_checkpoint(file)
        @test run_rows!(restored, rows, restored_norm).predictions == ev.predictions
    end
end

@testset "OTTT pairs each target with its own normalized spike trace" begin
    model = new_model("mixed", 123)
    fill!(model.input_weights, 0)
    fill!(model.weights, 0)
    fill!(model.tau, 0.1)
    norm = Dict("y_mean"=>zeros(4), "y_std"=>ones(4))
    rows = [Dict("session_id"=>"oracle", "split"=>"train", "segment_id"=>0,
        "frame_index"=>i, "eligible"=>true,"x"=>zeros(5),"y"=>fill(Float32(i),4)) for i in 0:2]
    run_rows!(model,rows,norm;train=true)
    # Bias 1.2 and tau 0.1 give spikes [0,1,0]. Independently calculated
    # trace values: [0, 1-exp(-0.2), exp(-0.2)*(1-exp(-0.2))].
    expected = 0.01 * ((1-exp(-0.2)) + 2exp(-0.2)*(1-exp(-0.2))) / 6
    @test all(isapprox.(model.weights, expected; rtol=1e-5))
end

function prepared_fixture()
    sessions = Any[]
    rows = fixture()
    for (sid,split) in [("s1","train"),("s2","validation"),("s3","test")]
        rs = filter(r->r["session_id"]==sid, rows)
        frames = [Dict("timestamp_ms"=>100r["frame_index"],"source_timestamp_ms"=>100r["frame_index"],
            "age_ms"=>0,"valid"=>true,"segment_id"=>0,"rejection_reasons"=>[],"x"=>r["x"]) for r in rs]
        examples = [Dict("frame_index"=>r["frame_index"],"y"=>r["y"],
            "history_indices"=>[r["frame_index"]-j for j in (5,10,20,50)],
            "target_timestamps_ms"=>[100r["frame_index"]+1000,100r["frame_index"]+5000]) for r in rs if r["eligible"]]
        push!(sessions,Dict("session_id"=>sid,"split"=>split,"seed"=>1,"frames"=>frames,"examples"=>examples))
    end
    Dict("schema_version"=>"anticipation-prepared-v1","sessions"=>sessions,
        "normalization"=>merge(target_normalization(rows),Dict("fit_split"=>"train","x_mean"=>zeros(5),"x_std"=>ones(5))),
        "feature_map_id"=>"anticipation-observed-gpu-v1",
        "feature_map"=>["vram","power","temperature","graphics_clock","memory_clock"],
        "target_names"=>["temperature_delta_1s_c","power_delta_1s_w","temperature_delta_5s_c","power_delta_5s_w"],
        "provenance"=>Dict("fixture"=>true))
end

@testset "Prepared fixture runs all seeds and budget exhaustion is explicit" begin
    mktempdir() do path
        prepared = joinpath(path,"prepared.json")
        AnticipationTrainer.write_json(prepared, prepared_fixture())
        report = campaign(prepared,joinpath(path,"runs"),120;epochs=2)
        @test length(report["runs"])==6
        @test all(r->r["status"]=="complete", report["runs"])
        @test all(r->r["epochs_completed"]==2, report["runs"])
        @test all(r->isfile(joinpath(path,"runs",r["predictions"])), report["runs"])
        curve = AnticipationTrainer.JSON3.read(read(joinpath(path,"runs","uniform-123","learning_curves.json"),String))
        @test all(c->haskey(c,:train_loss) && haskey(c,:validation_loss), curve)
        checkpoint = AnticipationTrainer.JSON3.read(read(joinpath(path,"runs","uniform-123","checkpoint.json"),String))
        @test get(checkpoint.metadata, :feature_map_id, nothing) == "anticipation-observed-gpu-v1"
        stopped = campaign(prepared,joinpath(path,"stopped"),0)
        @test all(r->r["status"]=="unfinished" && r["epochs_completed"]==0, stopped["runs"])
    end
end

@testset "JSON publication does not follow predictable staging links" begin
    for link_kind in (:symlink, :hardlink)
        mktempdir() do directory
            sentinel = joinpath(directory, "sentinel.json")
            write(sentinel, "source evidence")
            destination = joinpath(directory, "checkpoint.json")
            staging = destination * ".tmp"
            link_kind == :symlink ? symlink(sentinel, staging) : hardlink(sentinel, staging)
            AnticipationTrainer.write_json(destination, Dict("complete" => true))
            @test read(sentinel, String) == "source evidence"
            @test AnticipationTrainer.JSON3.read(read(destination, String)).complete
        end
    end
end
