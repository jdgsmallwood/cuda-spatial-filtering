# Kernel UDP capture on loopback

This runs a separate packet-sending process against Starspin's live kernel
backend. It exercises UDP reception, `recvmmsg`, the receive ring, reassembly,
GPU processing, and HDF5 output. It does not exercise a physical NIC or ibverbs.
The sender strips Ethernet/IP/UDP headers from the generated CODIF fixture and
advances frame counters on each repeat. Calibration is synthetic: unity gains
and zero delays, with source ID zero from `127.0.0.1`.

Build `starspin_1_8` using the reviewer guide, then open an isolated container:

```bash
docker run --rm --name starspin-loopback --network none --gpus all -it \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD":/workspace \
  -w /workspace starspin-joss-dev bash
```

Inside it, prepare a new output directory (use a new name for each run):

```bash
mkdir build/udp-live
cd build/udp-live
printf '%s\n' '{"frequency_plan":{"base_frequency_hz":0,"channel_bandwidth_hz":781250}}' > config.json
printf '%s\n' '{}' > gains.json
printf '%s\n' '{}' > eigenvalues.json
printf '%s\n' '{"0":0}' > delays.json
../apps/starspin_1_8 --packet-format codif --capture-backend kernel \
  -i lo -f 110 -c config.json -g gains.json -e eigenvalues.json \
  --delay-file delays.json -v visibilities.hdf5 \
  --accumulation-length 4 --obs-length 30 2>&1 | tee receiver.log
```

When the receiver prints `Setup completed. Ready to receive!`, run this from a
second host terminal:

```bash
docker exec -w /workspace starspin-loopback \
  python3 scripts/send_codif_fixture.py --packets 100000 --pps 5000
```

Use your image name and build directory if different. The sender is deliberately
restricted to loopback. `--pps` controls fixture-sized bursts, not precise
per-packet timing. Compare sent counts with receiver `Received`, `Processed`,
`Missing`, `Discarded`, and `NICDrops`; inspect `vis_missing_nums` before using
output. Sender success alone does not prove reception, and shutdown may leave
an incomplete final processing block.

Validated locally with `starspin-joss-dev` and `build_joss_clean`: 30,000 packets
at approximately 5,000 packets/s, all received and processed, zero reported
missing/discarded/NIC drops, and four finite, nonzero visibility integrations.
Logs and output are under `build_joss_clean/udp-synthetic-live/`. This is a socket-path
smoke test, not a numerical parity or sustained-throughput result. Increase rate
and duration separately; loopback cannot establish physical-NIC performance.
