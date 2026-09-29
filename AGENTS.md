# Live Starspin observations

At the start of every live observing session, verify the FPGA boot state and
the delay solution before judging correlator performance or data quality.
Consecutive captures within one observing session reuse the verified boot and
delay solution; do not reboot between them just because the channel range or
binary changes.
If the current boot state or matching delay calibration cannot be verified,
reboot the selected FPGAs, run full `init-tune` at eight channels, solve fresh
packet delays, and confirm delay graph closure. Retune to the requested wider
channel range only after the delay solve. Read back the channel range on every
board before capture. Use the delay file produced for that boot state.

Do not infer a CPU, GPU, or receive-ring bottleneck from a run with unverified
FPGA state or delays. Record the boot, tuning, delay file, and closure result
with the capture artifacts.
