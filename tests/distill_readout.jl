using Test
include(joinpath(@__DIR__, "..", "tools", "distill_readout.jl"))
const D = DistillReadout

@testset "shipped Distill readout" begin
    directory = only(ARGS)
    result = D.check_replay(directory)
    @test result.ticks == 10
    @test result.scores_checked == 30
    model, _ = D.shipped_readout()
    row = (stim=vcat(Float32[0.1, 0.25, 0.5, 0.3, 0.6], zeros(Float32, 11)),
           spikes=[0, 15], missing=String[])
    batch = D.mapped_batch(row)
    @test size(batch.spikes) == (16, 1)
    @test batch.targets == Float32[0.625, 0.5, 0.25]
    @test D.model_step(model, batch).logits == model.weights[:,1] + model.weights[:,16]
    checkpoint = D.JSON3.read(read(joinpath(D.BANK, "snn_model.json")))
    for (index, neuron) in enumerate(checkpoint.neurons)
        single = D.mapped_batch(merge(row, (spikes=[index-1],)))
        expected = round.(Int, 256 .* neuron.output_weights, RoundNearestTiesAway) ./ 256
        @test D.model_step(model, single).logits == expected
    end
    @test isnothing(D.mapped_batch(merge(row, (missing=["power_w"],))).targets)
    @test_throws ErrorException D.mapped_batch(merge(row, (spikes=[0, 0],)))
    @test_throws ErrorException D.mapped_batch(merge(row, (spikes=[16],)))
    # Exercise the real update interface on private model copies and a synthetic tick.
    for rule in (:eprop, :ottt)
        scratch = (weights=copy(model.weights),)
        loss(o) = sum(abs2, o.logits .- batch.targets)
        _, state = D.SynapticDistill.train_step!(scratch, batch, loss, D.model_step; rule)
        @test isfinite(state.loss)
        @test all(isfinite, scratch.weights)
        @test scratch.weights != model.weights
    end
    mktempdir() do tmp
        cp(joinpath(directory, "manifest.json"), joinpath(tmp, "manifest.json"))
        write(joinpath(tmp, "trace.jsonl"), "{}\n")
        @test_throws ErrorException D.check_replay(tmp)
    end
end
