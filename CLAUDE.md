# Project conventions — Gonzalez-diagnostics

## Long-running commands: always `systemd-run --user`

Any Julia run (simulation, plot script, analysis) that may take >5 s **must** be
launched as a transient systemd user unit, not a foreground shell. See
`~/.claude/skills/systemd-task/SKILL.md` for full reference.

### Naming convention

`junyi-task-julia-Gonzalez-diagnostics-<tag>` where `<tag>` matches the run's
`suffix` parameter or the plot script name. Examples:

- `junyi-task-julia-Gonzalez-diagnostics-bpmesh40k` — full simulation
- `junyi-task-julia-Gonzalez-diagnostics-fsdensity` — diagnostic plot script
- `junyi-task-julia-Gonzalez-diagnostics-smoketest` — short integration test

### Standard launch template

```bash
systemd-run --user \
    --unit=junyi-task-julia-Gonzalez-diagnostics-<tag> \
    --working-directory="$PWD" \
    --setenv=JULIA_NUM_THREADS=8 \
    julia --project=. <script.jl> [args...]
```

### Inspect

```bash
journalctl --user -u junyi-task-julia-Gonzalez-diagnostics-<tag>.service -f --no-pager
journalctl --user -u junyi-task-julia-Gonzalez-diagnostics-<tag>.service --no-pager -n 50
systemctl --user status junyi-task-julia-Gonzalez-diagnostics-<tag>.service --no-pager -l
```

### Kill

```bash
systemctl --user kill junyi-task-julia-Gonzalez-diagnostics-<tag>.service
systemctl --user reset-failed junyi-task-julia-Gonzalez-diagnostics-<tag>.service
```

### Forbidden

- `julia --project=. main_Gonzalez.jl …` directly in `Bash` tool. Reserve plain
  invocation only for `julia -e '<one-liner>'` style probes under 5 s.
- Backgrounded shell jobs (`&`, `nohup`) — not visible to systemd, no journald
  capture.

## Crash-safe CSV output

`main_Gonzalez.jl` streams `conservation_history_*.csv` per step and
`particle_snapshots_*.csv` per snapshot step (every 25 steps). Killed runs
still leave usable data through the last completed step — no need to wait for
end-of-run write.

## Mesh

Anisotropic non-uniform breakpoints `bp1` / `bp2` are configured in each
`parameters_*.jl` preset. CLI overrides are scalars only; edit the preset file
to change `bp1` / `bp2`.

## Output suffix

Set via `suffix` field in the preset. Drives every output filename
(`conservation_history_<suffix>.csv`, `fs_snapshot_<suffix>_step####.csv`,
`dashboard_<suffix>.png`, …). Pick a tag that includes mesh, particle count,
and solver knobs if they matter for the comparison being run.
