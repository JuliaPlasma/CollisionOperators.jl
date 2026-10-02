# GPU runtime image for Runpod pods: Julia + every dependency of main.jl
# (including CUDA.jl and its toolkit artifacts) installed and precompiled at
# build time, so a pod goes from boot to the first time step without Pkg work.
# On top of that, a root-only dev environment (neovim + plugins, zsh, yazi, …)
# for editing and inspecting runs on pods reachable only via the SSH proxy.
#
# Build (context = repository root):
#   docker build -f docker/Dockerfile -t ghcr.io/junyixu/collisionoperators .
#
# Run inside the pod (JULIA_PROJECT already points at the baked environment —
# do NOT pass --project=., the repository root has no Manifest):
#   cd /root/CollisionOperators.jl && git pull
#   julia main.jl parameters_LB_sq_d04.jl --use_gpu=true ...

# ---- dev tools (build stage) -------------------------------------------------
# Same Ubuntu release as runpod/base, so the binaries and compiled treesitter
# parsers run unchanged in the final stage. Built as root (HOME=/root): the
# configs and plugin caches are copied to the same paths in the final stage.
FROM ubuntu:24.04 AS devtools
ARG NVIM_VERSION=v0.12.5
ARG PARSERS="bash c diff html lua luadoc markdown markdown_inline query vim vimdoc python yaml latex typst"
ENV DEBIAN_FRONTEND=noninteractive HOME=/root
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git gcc make libc6-dev nodejs npm unzip zsh \
 && npm i -g tree-sitter-cli \
 && curl -fsSL https://github.com/neovim/neovim/releases/download/${NVIM_VERSION}/nvim-linux-x86_64.tar.gz | tar -xz -C /opt \
 && mv /opt/nvim-linux-x86_64 /opt/nvim
# extra CLI tools: eza / atuin / yazi from release binaries
RUN set -eux; cd /tmp; B=/opt/bin; mkdir -p $B; \
    base() { echo "https://github.com/$1/releases/latest/download/$2"; }; \
    curl -fsSL "$(base eza-community/eza eza_x86_64-unknown-linux-gnu.tar.gz)" | tar -xz -C $B; \
    curl -fsSL "$(base atuinsh/atuin atuin-x86_64-unknown-linux-gnu.tar.gz)" | tar -xz -C /tmp; \
    cp "$(find /tmp -name atuin -type f | head -1)" $B/atuin; \
    curl -fsSL -o y.zip "$(base sxyazi/yazi yazi-x86_64-unknown-linux-gnu.zip)"; \
    unzip -j y.zip '*/yazi' '*/ya' -d $B; chmod +x $B/*
ENV PATH=/opt/nvim/bin:/opt/bin:/root/.local/bin:$PATH
# Configs come straight from public GitHub repos. The *_REV args are cache
# busters only: the Docker workflow passes each branch's current commit, so a
# config push invalidates its layer (and everything after it in this stage).
ARG NVIM_BRANCH=feat/nvim-full-flag
ARG NVIM_REV=unset
RUN echo "nvim-config ${NVIM_BRANCH} @ ${NVIM_REV}" \
 && git clone --depth=1 --branch "${NVIM_BRANCH}" https://github.com/junyixu/nvim-config.git ~/.config/nvim \
 && git -C ~/.config/nvim rev-parse HEAD > ~/.config/nvim-rev \
 && rm -rf ~/.config/nvim/.git
ARG ZSH_BRANCH=main
ARG ZSH_REV=unset
RUN echo "zsh-config ${ZSH_BRANCH} @ ${ZSH_REV}" \
 && git clone --depth=1 --branch "${ZSH_BRANCH}" https://github.com/junyixu/zsh-config-public.git ~/.config/zsh \
 && git -C ~/.config/zsh rev-parse HEAD > ~/.config/zsh-rev \
 && rm -rf ~/.config/zsh/.git
ARG YAZI_BRANCH=main
ARG YAZI_REV=unset
RUN echo "yazi-config ${YAZI_BRANCH} @ ${YAZI_REV}" \
 && git clone --depth=1 --branch "${YAZI_BRANCH}" https://github.com/junyixu/yazi-config-public.git ~/.config/yazi \
 && git -C ~/.config/yazi rev-parse HEAD > ~/.config/yazi-rev \
 && rm -rf ~/.config/yazi/.git
RUN nvim --headless "+Lazy! install" +qa \
 && nvim --headless -c "lua require('nvim-treesitter').install(vim.split('${PARSERS}', ' ')):wait(900000)" -c qa
# blink.cmp fetches its prebuilt fuzzy lib on first load. The download writes `version`=v0.0.0 first and the real
# tag only after the checksum passes, so wait for that (not just for the .so) before nvim exits.
RUN for i in 1 2 3; do \
      nvim --headless \
        -c "lua require('lazy').load({ plugins = { 'blink.cmp' } })" \
        -c "lua local f = vim.fn.stdpath('data') .. '/lazy/blink.cmp/target/release/version'; print(vim.wait(120000, function() local h = io.open(f); if not h then return false end; local v = h:read('*a'); h:close(); return v ~= '' and v ~= 'v0.0.0' end, 200))" \
        -c "lua vim.wait(1000)" \
        -c qa 2>&1; \
      v=$(cat ~/.local/share/nvim/lazy/blink.cmp/target/release/version 2>/dev/null); \
      [ -n "$v" ] && [ "$v" != v0.0.0 ] && break; \
      echo "blink fuzzy download attempt $i failed" >&2; \
    done; \
    v=$(cat ~/.local/share/nvim/lazy/blink.cmp/target/release/version); echo "blink fuzzy lib version: $v"; \
    [ "$v" != v0.0.0 ] && test -f ~/.local/share/nvim/lazy/blink.cmp/target/release/libblink_cmp_fuzzy.so
RUN echo 'for f in ~/.config/zsh/zshrc.d/*.zsh(N); do source "$f"; done && typeset -U path PATH' > ~/.zshrc
# first interactive start clones zinit + every plugin/snippet, so bake it here
RUN TERM=xterm-256color zsh -ic 'exit' </dev/null
# p10k downloads gitstatusd on the first prompt; fetch it at build time instead
RUN ~/.local/share/zinit/plugins/romkatv---powerlevel10k/gitstatus/install
# slim: drop git history and per-plugin test dirs (no :Lazy update / zinit update inside the image; rebuild to update)
# blink.cmp keeps its .git: it runs `git describe --tags` to pick the matching prebuilt fuzzy lib
RUN find ~/.local/share/nvim/lazy ~/.local/share/zinit -path '*/lazy/blink.cmp/.git' -prune -o -name .git -prune -exec rm -rf {} + \
 && find ~/.local/share/nvim/lazy ~/.local/share/zinit/plugins -mindepth 2 -maxdepth 2 -type d \( -name test -o -name tests \) -prune -exec rm -rf {} +

# ---- runtime image -------------------------------------------------------------
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

# Everything below sits after the Julia depot layer, so changing the dev tools
# or the shell environment never rebuilds the depot.

# Pods expose fewer vCPUs than the host has cores, but OpenBLAS sizes its pool
# by host cores; the oversubscribed threads spin and starve the GPU. One thread
# was fastest for the Anderson least-squares on a 12-vCPU RTX 4090 pod.
ENV OPENBLAS_NUM_THREADS=1

# Dev tools: runtime packages here, the built nvim / CLI tools / configs from the
# devtools stage. Root only — sshd and authorized_keys stay with runpod/base.
RUN apt-get update && apt-get install -y --no-install-recommends \
      git zsh tmux fzf zoxide direnv bat ripgrep fd-find file less ncurses-term kitty-terminfo \
 && ln -sf /usr/bin/batcat /usr/local/bin/bat && ln -sf /usr/bin/fdfind /usr/local/bin/fd \
 && usermod -s /usr/bin/zsh root \
 && rm -rf /var/lib/apt/lists/*
COPY --from=devtools /opt/nvim /opt/nvim
COPY --from=devtools /opt/bin/ /usr/local/bin/
RUN ln -s /opt/nvim/bin/nvim /usr/local/bin/nvim
COPY --from=devtools /root/.config /root/.config
COPY --from=devtools /root/.local /root/.local
COPY --from=devtools /root/.zshrc /root/.zshrc
COPY --from=devtools /root/.cache /root/.cache

# Image ENV reaches only PID 1, not SSH sessions: replay it for login shells
# (/etc/profile.d) and for every zsh, including `ssh pod 'cmd'` (/etc/zsh/zshenv;
# zsh on Ubuntu does not read /etc/profile).
RUN { echo "export JULIA_DEPOT_PATH='$JULIA_DEPOT_PATH'"; \
      echo "export JULIA_PROJECT='$JULIA_PROJECT'"; \
      echo "export JULIA_CPU_TARGET='$JULIA_CPU_TARGET'"; \
      echo "export OPENBLAS_NUM_THREADS='$OPENBLAS_NUM_THREADS'"; \
      echo 'case ":$PATH:" in *:/opt/julia/bin:*) ;; *) export PATH="/opt/julia/bin:$PATH" ;; esac'; \
    } > /etc/profile.d/julia.sh \
 && echo '[ -r /etc/profile.d/julia.sh ] && emulate sh -c ". /etc/profile.d/julia.sh"' >> /etc/zsh/zshenv

COPY . /root/CollisionOperators.jl
WORKDIR /root/CollisionOperators.jl
