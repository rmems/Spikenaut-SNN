# syntax=docker/dockerfile:1
#
# Cloud Agent base image for Spikenaut-SNN.
#
# Python is the latest stable interpreter (the 3.14 series, currently 3.14.7),
# pinned at the minor level so patch releases roll forward without a Dockerfile
# edit. This is the *development* interpreter only -- CI still pins 3.11 on
# purpose (see .github/workflows/ci.yml), and the tools parse back to 3.8.
# Hub tag python:3.14-slim-bookworm is current (verified 2026-09); do not
# downgrade to 3.12.
FROM python:3.14-slim-bookworm

# git and curl are required in a fully custom Cloud Agent image; build-essential
# and pkg-config supply the C toolchain the Rust dependencies link against, and
# sudo matches the Cloud Agent convention of a passwordless non-root user.
# bubblewrap and libseccomp2 are what ci.yml installs for the Hermes tests.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bubblewrap \
        build-essential \
        ca-certificates \
        curl \
        git \
        libseccomp2 \
        pkg-config \
        sudo \
    && rm -rf /var/lib/apt/lists/*

# Cloud Agent tooling (git checkout, terminals, environment `install`) expects
# an `ubuntu` user with passwordless sudo. NOPASSWD is required: the Cloud
# Agent convention has no interactive TTY for a sudo password, and the
# harness must be able to install packages and mutate the image as root.
# Do not remove NOPASSWD or the agent cannot complete setup.
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
# The Cloud Agent install/terminal shells do not inherit this image's ENV PATH,
# so the rustup proxies are also symlinked onto a PATH dir every shell already
# has and exported from a login profile. Without this, `install` runs before
# PATH is set and fails with `rustup: command not found`.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --no-modify-path --default-toolchain 1.98.1 --profile minimal -c rustfmt -c clippy \
    && chown -R ubuntu:ubuntu "$RUSTUP_HOME" "$CARGO_HOME" \
    && ln -sf /usr/local/cargo/bin/* /usr/local/bin/ \
    && printf 'export RUSTUP_HOME=%s\nexport CARGO_HOME=%s\nexport PATH=%s/bin:$PATH\n' \
        "$RUSTUP_HOME" "$CARGO_HOME" "$CARGO_HOME" > /etc/profile.d/rust.sh

USER ubuntu
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Julia for tools/anticipation, pinned like ci.yml's anticipation-julia job.
# juliaup installs per user, so the launchers are also linked into /usr/local/bin
# for shells that don't read the login profile.
RUN curl --proto '=https' --tlsv1.2 -fsSL https://install.julialang.org \
        | sh -s -- --yes --default-channel 1.12.7 \
    && sudo ln -sf /home/ubuntu/.juliaup/bin/julia /home/ubuntu/.juliaup/bin/juliaup /usr/local/bin/
