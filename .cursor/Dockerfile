# syntax=docker/dockerfile:1
#
# Cloud Agent base image for Spikenaut-SNN.
#
# Python is the latest stable interpreter (the 3.14 series, currently 3.14.7),
# pinned at the minor level so patch releases roll forward without a Dockerfile
# edit. This is the *development* interpreter only -- CI still pins 3.11 on
# purpose (see .github/workflows/ci.yml), and the tools parse back to 3.8.
FROM python:3.14-slim-bookworm

# git and curl are required in a fully custom Cloud Agent image; build-essential
# and pkg-config supply the C toolchain the Rust dependencies link against, and
# sudo matches the Cloud Agent convention of a passwordless non-root user.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        pkg-config \
        sudo \
    && rm -rf /var/lib/apt/lists/*

# Cloud Agent tooling (git checkout, terminals) expects an `ubuntu` user.
RUN useradd --create-home --shell /bin/bash ubuntu \
    && echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu

# Rust is installed system-wide, then handed to the ubuntu user. rust-toolchain.toml
# pins the exact toolchain (1.98.1 + rustfmt + clippy) and rustup honors that
# override inside /workspace, so this preinstall only needs to match that pin to
# keep the first build offline and fast.
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --no-modify-path --default-toolchain 1.98.1 --profile minimal -c rustfmt -c clippy \
    && chown -R ubuntu:ubuntu "$RUSTUP_HOME" "$CARGO_HOME"

USER ubuntu
