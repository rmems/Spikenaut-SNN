#!/usr/bin/env julia
# SPDX-License-Identifier: MIT OR Apache-2.0

module DistillReadout

using JSON3, SHA, SynapticDistill

const BANK = joinpath(@__DIR__, "..", "dataset", "merged_v2")
const HEADS = ["comfort", "temp", "power"]

function checked_artifact(provenance, name)
    bytes = read(joinpath(BANK, name))
    bytes2hex(sha256(bytes)) == provenance.artifacts[name].sha256 ||
        error("promoted artifact digest mismatch: $name")
    return bytes
end

function shipped_readout()
    provenance = JSON3.read(read(joinpath(BANK, "provenance.json")))
    checked_artifact(provenance, "snn_model.json")
    bytes = checked_artifact(provenance, "parameters_output_weights.mem")
    words = parse.(UInt16, split(String(bytes)); base=16)
    length(words) == 48 || error("expected 16 neuron rows of 3 signed Q8.8 words")
    # Each group of three consecutive words is one presynaptic column.
    weights = reshape(Float32.(reinterpret(Int16, words)) ./ 256f0, 3, 16)
    digest = provenance.artifacts["snn_model.json"].sha256
    return (; weights), "sha256:" * digest
end

function mapped_batch(row)
    stim = row.stim
    length(stim) == 16 && all(v -> v isa Real && !(v isa Bool) && 0 <= v <= 1, stim) &&
        all(iszero, stim[6:16]) || error("invalid live-five analog frame")
    spikes = zeros(Float32, 16, 1)
    for id in row.spikes
        id isa Integer && !(id isa Bool) && 0 <= id < 16 || error("invalid hidden neuron ID")
        iszero(spikes[id+1, 1]) || error("duplicate hidden neuron ID")
        spikes[id+1, 1] = 1f0
    end
    # Temperature zero is also the v3 dropout sentinel. Withhold labels for
    # unknown temperature or power while retaining the tick for frozen readout.
    temp, power = Float32(stim[3]), Float32(stim[2])
    targets = if temp > 0 && !("gpu_temp_c" in row.missing || "power_w" in row.missing)
        Float32[clamp(1f0 - 0.5f0*temp - 0.5f0*power, 0f0, 1f0), temp, power]
    end
    return SpikeBatch(spikes, nothing, targets)
end

model_step(model, batch::SpikeBatch) = (logits=vec(model.weights * batch.spikes),)

function check_replay(directory)
    model, digest = shipped_readout()
    manifest = JSON3.read(read(joinpath(directory, "manifest.json")))
    manifest.schema_version == "spikenaut.replay-manifest.v1" &&
        manifest.trace_schema == "spikenaut.replay-trace.v1" || error("unsupported replay schema")
    manifest.model.id == "merged_v2" && manifest.model.checkpoint_digest == digest &&
        manifest.model.numeric_format == "q8.8-fixed-point" &&
        manifest.model.feature_map_id == "spikenaut.feature-map.live-exp-025.v1" &&
        collect(manifest.output.vocabulary) == HEADS || error("replay does not name the shipped readout")
    trace = read(joinpath(directory, "trace.jsonl"))
    "sha256:" * bytes2hex(sha256(trace)) == manifest.trace_sha256 || error("trace digest mismatch")
    ticks = labeled = 0
    for line in eachline(IOBuffer(trace))
        row = JSON3.read(line)
        row.step == ticks || error("unexpected trace step at tick $ticks")
        batch = mapped_batch(row)
        model_step(model, batch).logits == row.scores || error("readout mismatch at tick $ticks")
        ticks += 1
        labeled += !isnothing(batch.targets)
    end
    ticks > 0 && ticks == manifest.input.n_steps || error("replay length mismatch")
    return (; ticks, labeled, scores_checked=3*ticks, checkpoint=digest, trace=manifest.trace_sha256)
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("usage: distill_readout.jl REPLAY_DIRECTORY")
    println(DistillReadout.JSON3.write(DistillReadout.check_replay(only(ARGS))))
end
