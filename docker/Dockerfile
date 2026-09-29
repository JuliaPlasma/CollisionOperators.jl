# GPU runtime image for Runpod pods: Julia + every dependency of main.jl
# (including CUDA.jl and its toolkit artifacts) installed and precompiled at
# build time, so a pod goes from boot to the first time step without Pkg work.
#
# Build (context = repository root):
#   docker build -f docker/Dockerfile -t ghcr.io/junyixu/collisionoperators .
#
# Run inside the pod (JULIA_PROJECT already points at the baked environment —
# do NOT pass --project=., the repository root has no Manifest):
#   cd /root/CollisionOperators.jl && git pull
#   julia main.jl parameters_LB_sq_d04.jl --use_gpu=true ...

# Runpod's base image: its default CMD starts sshd from the PUBLIC_KEY env var.
FROM runpod/base:1.0.2-ubuntu2404

ARG JULIA_VERSION=1.12.7

# Precompiled package images are CPU-specific. Build for the same multi-target
# set as the official Julia binaries, so the images stay valid on whatever host
# CPU the pod lands on instead of being recompiled at first `using`.
ENV JULIA_DEPOT_PATH=/opt/julia-depot \
    JULIA_PROJECT=/opt/collisionops-env \
    JULIA_CPU_TARGET="generic;sandybridge,-xsaveopt,clone_all;haswell,-rdrnd,base(1);x86-64-v4,-rdrnd,base(1)" \
    PATH=/opt/julia/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends unzip \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://rclone.org/install.sh | bash \
    && curl -fsSL "https://julialang-s3.julialang.org/bin/linux/x64/${JULIA_VERSION%.*}/julia-${JULIA_VERSION}-linux-x86_64.tar.gz" \
    | tar -xz -C /opt \
    && mv "/opt/julia-${JULIA_VERSION}" /opt/julia \
    && julia --version

# Dependencies first, so a code-only change reuses this (slow) layer.
COPY docker/Project.toml docker/Manifest.toml docker/LocalPreferences.toml $JULIA_PROJECT/
RUN julia -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(; strict = true)' \
    && julia -e 'using CUDA, CairoMakie, Mantis' \
    && find $JULIA_DEPOT_PATH/artifacts -name 'libcudart.so.*' | grep -q . \
    && chmod -R a+rX $JULIA_DEPOT_PATH

COPY . /root/CollisionOperators.jl
WORKDIR /root/CollisionOperators.jl
