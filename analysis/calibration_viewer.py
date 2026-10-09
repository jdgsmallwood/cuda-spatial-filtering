"""Interactive evidence viewer for the 400–439 Starspin calibration run.

Run with: /data/BLACKMESA_1/sma075/rfi-monitor/.venv/bin/marimo run analysis/calibration_viewer.py
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    from pathlib import Path

    import h5py
    import marimo as mo
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    import numpy as np
    import pandas as pd
    from scipy.optimize import least_squares

    from calibration_viewer_utils import (
        choose_capture, dirty_image, gain_key, load_report,
        read_mean_visibility, score_rows, selected_frequency, source_offsets,
    )

    return (Circle, Path, choose_capture, dirty_image, gain_key, h5py, json,
            least_squares, load_report, mo, np, pd, plt, read_mean_visibility, score_rows,
            selected_frequency, source_offsets)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Starspin calibration evidence

    Select a calibration `report.json` at the top. This viewer shows held-out
    calibration scores, RFI masks, FPGA delay checks, candidate gains,
    direct single-channel images, and a cached all-fine-channel YY image.
    **The saved gains are candidates, not accepted
    calibration weights.** A bright image peak alone does not validate them;
    compare held-out times and baselines first.
    """)
    return


@app.cell
def _(Path, mo):
    _default = Path("/data/BLACKMESA_1/sma075/lambda-commissioning/runs/"
                    "rfi_cal_400_439_20261003/comparison_corrected_final/report.json")
    report_browser = mo.ui.file_browser(initial_path=_default.parent,
                                        filetypes=[".json"], multiple=False,
                                        restrict_navigation=False,
                                        label="Choose a calibration report")
    report_text = mo.ui.text(value=str(_default), full_width=True,
                             label="Or report.json path")
    mo.vstack([report_browser, report_text])
    return report_browser, report_text


@app.cell
def _(Path, load_report, mo, report_browser, report_text):
    _picked = report_browser.path(index=0) if report_browser.value else report_text.value
    report_path = Path(_picked).expanduser()
    mo.stop(not report_path.is_file(), mo.callout("Choose an existing report.json.", kind="warn"))
    report = load_report(report_path)
    mo.stop("results" not in report, mo.callout("This is not a calibration comparison report.", kind="warn"))
    return report, report_path


@app.cell
def _(mo, report):
    _loss = 100 * report["packet_loss_fraction"]
    mo.md(f"""
    **Capture:** `{report['capture']}`  
    **Time:** {report['utc_range'][0]} – {report['utc_range'][1]} UTC  
    **Packet loss:** {_loss:.5f}% · **RFI mask retained:**
    {100 * report['time_frequency_retained_fraction']:.2f}% of sampled cells  
    **Fit:** {report['baseline_minimum_m']} m minimum baseline, antenna(s)
    {report['excluded_antenna_ids']} excluded; final third of time and 20% of
    baselines held out. `{report['status']}`.
    """)
    return


@app.cell
def _(mo, report):
    _freqs = report["frequencies_hz"]
    frequency_picker = mo.ui.dropdown(
        options={f"{_f / 1e6:.5f} MHz (coarse {_i + 400})": _i
                 for _i, _f in enumerate(_freqs)},
        value=f"{_freqs[14] / 1e6:.5f} MHz (coarse 414)",
        label="Coarse-channel centre for detail/image")
    pol_picker = mo.ui.dropdown(options=["YY", "XX"], value="YY", label="Polarization")
    method_picker = mo.ui.dropdown(
        options={"Multi-source model": "A_multisource",
                 "Pictor A tracked": "B_Pictor", "Fornax A tracked": "C_Fornax"},
        value="Multi-source model", label="Gain candidate")
    stage_picker = mo.ui.dropdown(options=["amplitude", "phase"],
                                  value="amplitude", label="Gain stage")
    mo.hstack([frequency_picker, pol_picker, method_picker, stage_picker],
              justify="start", gap=1)
    return frequency_picker, pol_picker, method_picker, stage_picker


@app.cell
def _(mo):
    show_source_labels = mo.ui.checkbox(
        value=False, label="Show source names on images (markers stay visible)")
    show_source_labels
    return show_source_labels


@app.cell
def _(Path, mo, np):
    _default = Path("/data/BLACKMESA_1/sma075/cuda-spatial-filtering/analysis/"
                    "results/yy_fine_delays")
    fine_dir_text = mo.ui.text(value=str(_default), full_width=True,
                               label="Optional all-fine YY candidate directory")
    fine_dir_text
    return fine_dir_text


@app.cell
def _(Path, fine_dir_text, np):
    _dir = Path(fine_dir_text.value).expanduser()
    fine_report = None
    fine_data = None
    if (_dir / "report.json").is_file() and (_dir / "fine_yy_candidates.npz").is_file():
        import json as _json
        fine_report = _json.loads((_dir / "report.json").read_text())
        with np.load(_dir / "fine_yy_candidates.npz") as _z:
            fine_data = {k: _z[k] for k in _z.files}
    return fine_data, fine_report


@app.cell(hide_code=True)
def _(fine_data, fine_report, mo, np, pd, plt):
    mo.stop(fine_data is None, mo.md("All-fine YY results are not loaded."))
    _v = fine_report["delay_validation"]
    _intro = mo.md(f"""
    ### All-fine YY and delay check

    {fine_report['fitted_fine_bins']} retained fine bins fitted. A single delay
    per antenna improved median held-time correlation from
    **{_v['unchanged_centre_gain_held_time']:.3f}** to
    **{_v['with_delay_held_time']:.3f}**, and held-baseline correlation from
    **{_v['unchanged_centre_gain_held_baselines']:.3f}** to
    **{_v['with_delay_held_baselines']:.3f}**. This is a diagnostic gain, not
    yet a validated science calibration. Slopes vary among coarse channels.
    """)
    _fig, _axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    _freq = fine_data["frequency_hz"] / 1e6
    _score = fine_data["delay_validation_scores"]
    for _trial, _name in enumerate(("Centre gain", "With fitted delay")):
        _axes[0].plot(_freq, _score[:, _trial, 0], ".", ms=2, label=_name)
        _axes[1].plot(_freq, _score[:, _trial, 1], ".", ms=2, label=_name)
    for _ax, _label in zip(_axes, ("Held-out time", "Held-out baselines")):
        _ax.set(ylabel=f"{_label} correlation", ylim=(-.3, 1.05))
        _ax.grid(alpha=.25)
        _ax.legend()
    _axes[-1].set_xlabel("Fine frequency (MHz)")
    _table = pd.DataFrame({"Antenna": fine_data["antenna_ids"],
                           "Delay vs antenna 1 (ns)": np.round(fine_data["delay_s"] * 1e9, 1),
                           "Across-band MAD (ns)": np.round(fine_data["delay_mad_s"] * 1e9, 1)})
    mo.vstack([_intro, _fig, mo.ui.table(_table)])
    return


@app.cell(hide_code=True)
def _(fine_data, json, least_squares, mo, np, pd, plt, report_path):
    mo.stop(fine_data is None)
    _config = json.loads((report_path.parent.parent / "starspin_config.json").read_text())
    _board = {int(antenna): int(fpga)
              for fpga, mapping in _config["antenna_positions"].items()
              for antenna in mapping}
    _position = {int(antenna): (v["east"], v["north"])
                 for mapping in _config["antenna_positions"].values()
                 for antenna, v in mapping.items()}
    _ids = fine_data["antenna_ids"]
    _delay = fine_data["delay_s"] * 1e9
    _fpga = np.array([_board[int(antenna)] for antenna in _ids])
    _medians = {int(board): float(np.nanmedian(_delay[_fpga == board]))
                for board in np.unique(_fpga)}
    _residual = np.array([d - _medians[int(board)]
                          for d, board in zip(_delay, _fpga)])
    _east = np.array([_position[int(a)][0] for a in _ids])
    _north = np.array([_position[int(a)][1] for a in _ids])
    _design = np.column_stack([(_fpga == board).astype(float) for board in range(4)]
                              + [_east, _north])
    _initial = np.linalg.lstsq(_design, _delay, rcond=None)[0]
    _fit = least_squares(lambda p: _design @ p - _delay, _initial,
                         loss="soft_l1", f_scale=20)
    _spatial_residual = _delay - _design @ _fit.x
    _fig, _axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    _colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")
    _records = []
    for _board_id, _color in enumerate(_colors):
        _select = _fpga == _board_id
        if not np.any(_select):
            continue
        _axes[0].scatter(_ids[_select], _delay[_select], s=55,
                         color=_color, edgecolor="black", linewidth=.5,
                         label=f"FPGA {_board_id}")
        _axes[1].scatter(_ids[_select], _residual[_select], s=55,
                         color=_color, edgecolor="black", linewidth=.5,
                         label=f"FPGA {_board_id}")
        _axes[2].scatter(_ids[_select], _spatial_residual[_select], s=55,
                         color=_color, edgecolor="black", linewidth=.5,
                         label=f"FPGA {_board_id}")
        _records.append({"FPGA": _board_id, "Antennas": int(_select.sum()),
                         "Median delay vs antenna 1 (ns)": round(_medians[_board_id], 1),
                         "Within-FPGA MAD (ns)": round(float(
                             1.4826 * np.nanmedian(np.abs(_residual[_select]))), 1)})
    _axes[0].set(xlabel="Antenna ID", ylabel="Delay vs antenna 1 (ns)",
                 title="Per-antenna YY phase-slope delay")
    _axes[1].set(xlabel="Antenna ID", ylabel="Delay minus FPGA median (ns)",
                 title="Variation within each FPGA")
    _axes[2].set(xlabel="Antenna ID", ylabel="Residual delay (ns)",
                 title="After FPGA offsets and position slope")
    _axes[1].axhline(0, color="black", lw=1)
    _axes[2].axhline(0, color="black", lw=1)
    for _ax in _axes:
        _ax.grid(alpha=.25)
        _ax.legend(fontsize=8)
    _reference_residual = float(_spatial_residual[_ids == 1][0])
    mo.vstack([mo.md(f"#### Delay grouped by FPGA\n\nAntenna 1 is fixed at 0 ns as the phase reference. The fitted east–west slope is **{_fit.x[4]:.2f} ns/m**. After allowing one offset per FPGA and an east/north position slope, antenna 1 is only **{_reference_residual:.1f} ns** from the trend. The spatial slope may absorb sky-model or phase-convention errors, so these are diagnostic values, not hardware delay settings."),
               _fig, mo.ui.table(pd.DataFrame(_records))])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Held-out calibration scores

    Curves score each fitted gain against the **same full point-source sky**.
    Solid lines use the final third of the observation, which was excluded
    from the fit. Dashed lines use baselines withheld from the fit. A score
    near 1 on both tests, across neighboring frequencies and time windows,
    would support a gain solution. Training scores alone are optimistic.
    """)
    return


@app.cell
def _(frequency_picker, mo, np, plt, pol_picker, report, score_rows, stage_picker):
    _pol = pol_picker.value
    _stage = stage_picker.value
    _fig, _axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                              constrained_layout=True)
    _colors = {"A_multisource": "tab:blue", "B_Pictor": "tab:orange",
               "C_Fornax": "tab:green"}
    for _method, _color in _colors.items():
        _rows = [r for r in score_rows(report, _pol, _stage) if r["method"] == _method]
        _x = np.array([r["frequency_hz"] / 1e6 for r in _rows])
        _axes[0].plot(_x, [r["common_sky_held_time"]["correlation"] for r in _rows],
                      ".-", color=_color, label=_method)
        _axes[1].plot(_x, [r["common_sky_held_baselines"]["correlation"] for r in _rows],
                      ".--", color=_color, label=_method)
    for _ax, _title in zip(_axes, ["Held-out time", "Held-out baselines"]):
        _ax.axvline(report["frequencies_hz"][frequency_picker.value] / 1e6,
                    color="black", alpha=.4)
        _ax.axhline(.6, color="gray", alpha=.45, lw=1)
        _ax.set(ylabel="Correlation with full sky", title=_title, ylim=(-.2, 1.05))
        _ax.grid(alpha=.25)
        _ax.legend(fontsize=8, ncol=3)
    _axes[-1].set_xlabel("Coarse-channel centre (MHz)")
    _fig
    return


@app.cell
def _(frequency_picker, method_picker, mo, pd, pol_picker, report, stage_picker):
    _f = report["frequencies_hz"][frequency_picker.value]
    _rows = [r for r in report["results"] if r["pol"] == pol_picker.value
             and r["stage"] == stage_picker.value and r["method"] == method_picker.value
             and abs(r["frequency_hz"] - _f) < 1]
    _table = pd.DataFrame([{
        "Frequency MHz": round(_f / 1e6, 5),
        "Training": round(r["train"]["correlation"], 3),
        "Held time": round(r["common_sky_held_time"]["correlation"], 3),
        "Held baselines": round(r["common_sky_held_baselines"]["correlation"], 3),
        "Clean train integrations": r["retained_train_integrations"],
        "Clean held integrations": r["retained_held_integrations"],
    } for r in _rows])
    mo.stop(_table.empty, mo.callout(
        "This centre frequency was rejected for insufficient clean samples.", kind="warn"))
    mo.ui.table(_table)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## RFI mask

    Orange cells are discarded before calibration; white cells were retained.
    The upper panel shows all 40 coarse channels and ten minutes. The lower
    panel expands the selected coarse channel to its 27 kept fine bins, so
    narrow lines and intermittent bursts are visible. The mask catches strong
    auto-power events; a stationary coherent contaminant can remain unflagged.
    """)
    return


@app.cell
def _(mo, report, report_path):
    _run = report_path.parent.parent
    _default = _run / "flags_corrected_v2/transient_mask.npz"
    mask_text = mo.ui.text(value=str(_default), full_width=True,
                           label="RFI mask path (change after copying to laptop)")
    mask_text
    return mask_text


@app.cell
def _(Path, mask_text, mo, np):
    _path = Path(mask_text.value).expanduser()
    mo.stop(not _path.is_file(), mo.callout("RFI mask file unavailable at this path.", kind="warn"))
    with np.load(_path) as _mask:
        mask_good = _mask["good"].astype(bool)
        mask_freq = _mask["frequency_hz"].astype(float)
    return mask_good, mask_freq


@app.cell
def _(frequency_picker, mask_freq, mask_good, np, plt, report):
    _index = frequency_picker.value
    _start = 27 * _index
    _stop = _start + 27
    _fig, _axes = plt.subplots(2, 1, figsize=(13, 6), constrained_layout=True)
    _coarse = (~mask_good).reshape(mask_good.shape[0], 40, 27).mean(axis=2)
    _im = _axes[0].imshow(_coarse, aspect="auto", origin="lower", cmap="magma",
                          vmin=0, vmax=1, extent=[400, 440, 0, 10])
    _axes[0].axvline(400 + _index + .5, color="cyan", lw=1)
    _axes[0].set(xlabel="Coarse channel", ylabel="Minutes from start",
                  title="Fraction of fine bins excluded")
    _fig.colorbar(_im, ax=_axes[0], label="Flagged fraction")
    _axes[1].imshow(~mask_good[:, _start:_stop], aspect="auto", origin="lower",
                    cmap="magma", vmin=0, vmax=1,
                    extent=[mask_freq[_start] / 1e6, mask_freq[_stop-1] / 1e6, 0, 10])
    _axes[1].set(xlabel="Fine frequency (MHz)", ylabel="Minutes from start",
                  title=f"Fine-bin flags for coarse {400 + _index}")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## FPGA delay verification

    The two independent solves before the corrected capture and the solve
    afterwards should agree pair by pair, with zero graph-closure residual.
    These sample offsets were **already applied during capture**. The image
    below uses the corrected visibilities and does not apply them a second time.
    """)
    return


@app.cell
def _(json, mo, pd, report_path):
    _run = report_path.parent.parent
    _names = ("Before A", "Before B", "After")
    _dirs = ("delay_highband", "delay_highband_repeat",
             "delay_highband_after_corrected")
    _records = []
    for _name, _directory in zip(_names, _dirs):
        _path = _run / _directory / "pairwise_delays.json"
        if _path.is_file():
            for _r in json.loads(_path.read_text()):
                _records.append({"Solve": _name, "FPGA pair": f"{_r['first']}–{_r['second']}",
                                 "Delay samples": _r["delay"],
                                 "Closure error": _r["closure_residual_samples"],
                                 "Votes": f"{_r['votes']}/{_r['total']}"})
    mo.stop(not _records, mo.callout("Pairwise delay reports unavailable.", kind="info"))
    mo.ui.table(pd.DataFrame(_records))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Candidate gains and example dirty image

    Select a complete integration from the sampled calibration times and an
    averaging window around it. The left image uses measured visibilities;
    the right divides them by the
    selected candidate gains. Both use the same baselines, with autos and
    baselines under 10 m excluded. **Held baselines** were not used to fit the
    gains and are the more useful visual check. Yellow markers show catalogue
    source directions above the horizon. The circle is the local horizon; the
    map includes the full visible hemisphere in east/north direction cosines.
    This is a direct zenith-centred Fourier dirty image, without CLEAN or a
    beam correction, so sidelobes are expected. Its absolute brightness is
    not calibrated.
    """)
    return


@app.cell
def _(choose_capture, mo, report, report_path):
    _default = choose_capture("", report_path, report)
    capture_text = mo.ui.text(value=str(_default), full_width=True,
                              label="Raw visibility HDF5 path (optional for scores/masks)")
    capture_text
    return capture_text


@app.cell
def _(capture_text, mo, np, report_path):
    _path = report_path.parent / "candidates.npz"
    mo.stop(not _path.is_file(), mo.callout("Candidate gain file unavailable.", kind="warn"))
    with np.load(_path) as _gain_file:
        gain_data = {k: _gain_file[k] for k in _gain_file.files}
    _n = len(gain_data["times_mjd"])
    time_picker = mo.ui.slider(start=0, stop=_n-1, value=_n//2, step=1,
                               label="Sampled integration index")
    baseline_picker = mo.ui.dropdown(options={"Held baselines": "held",
                                              "All usable baselines": "all"},
                                     value="Held baselines", label="Image baselines")
    average_picker = mo.ui.dropdown(
        options={"One integration": 1, "About 10 seconds": 5,
                 "About 30 seconds": 15, "About 60 seconds": 30},
        value="About 30 seconds", label="Image averaging")
    fine_bin_picker = mo.ui.slider(start=0, stop=26, value=13, step=1,
                                   label="Fine bin within selected coarse channel")
    gain_choice = mo.ui.dropdown(
        options={"Centre-bin candidate": "centre", "All-fine YY candidate": "fine",
                 "Single-delay YY model": "delay"},
        value="Centre-bin candidate", label="Image gain source")
    mo.vstack([mo.hstack([time_picker, baseline_picker, average_picker],
                        justify="start", gap=1),
               mo.hstack([fine_bin_picker, gain_choice], justify="start", gap=1)])
    return gain_data, time_picker, baseline_picker, average_picker, fine_bin_picker, gain_choice


@app.cell
def _(fine_bin_picker, fine_data, frequency_picker, gain_choice, gain_data,
          gain_key, method_picker, mo, np, plt, pol_picker, stage_picker):
    _key = gain_key(method_picker.value, stage_picker.value,
                    pol_picker.value, frequency_picker.value)
    mo.stop(_key not in gain_data, mo.callout(
        "No gain candidate exists at this frequency; choose another centre.", kind="warn"))
    _g = gain_data[_key]
    if gain_choice.value != "centre" and pol_picker.value == "YY" and fine_data is not None:
        _fine_index = frequency_picker.value * 27 + fine_bin_picker.value
        if gain_choice.value == "fine":
            _g = fine_data["gains"][_fine_index]
        else:
            _f = fine_data["frequency_hz"][_fine_index]
            _f0 = fine_data["frequency_hz"][frequency_picker.value * 27 + 13]
            _g = _g * np.exp(2j * np.pi * (_f - _f0) * fine_data["delay_s"])
        mo.stop(not np.all(np.isfinite(_g)), mo.callout(
            "No finite all-fine gain exists for this bin.", kind="warn"))
    _ids = gain_data["antenna_ids"]
    _fig, _axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True,
                              constrained_layout=True)
    _axes[0].plot(_ids, np.abs(_g), ".-")
    _axes[1].plot(_ids, np.angle(_g), ".-")
    _axes[0].set(ylabel="Gain amplitude (arbitrary units)")
    _axes[1].set(xlabel="Antenna ID", ylabel="Gain phase (rad)", ylim=(-3.3, 3.3))
    for _ax in _axes:
        _ax.grid(alpha=.3)
    _fig
    return


@app.cell
def _(Circle, Path, average_picker, baseline_picker, capture_text, dirty_image, fine_bin_picker,
          fine_data, frequency_picker, gain_choice, gain_data, gain_key, json,
          mask_freq, mask_good, method_picker, mo, np,
          plt, pol_picker, read_mean_visibility, report, report_path,
          show_source_labels, source_offsets, stage_picker, time_picker):
    _capture = Path(capture_text.value).expanduser()
    mo.stop(not _capture.is_file(), mo.callout(
        "Set a readable raw HDF5 path to display example images.", kind="info"))
    _fi = frequency_picker.value
    _key = gain_key(method_picker.value, stage_picker.value, pol_picker.value, _fi)
    mo.stop(_key not in gain_data, mo.callout("No gain candidate at this frequency.", kind="warn"))
    _integration = report["used_integrations"][time_picker.value]
    _channel = 27 * _fi + fine_bin_picker.value
    mo.stop(gain_choice.value != "centre" and
            (fine_data is None or pol_picker.value != "YY"), mo.callout(
                "All-fine candidates are available for YY after loading the fine result directory.",
                kind="warn"))
    _n_average = average_picker.value
    _lo = max(0, time_picker.value - _n_average // 2)
    _hi = min(len(report["used_integrations"]), _lo + _n_average)
    _samples = np.asarray(report["used_integrations"][_lo:_hi], dtype=int)
    _samples = _samples[mask_good[_samples, _channel]]
    mo.stop(len(_samples) == 0, mo.callout(
        "All nearby integrations are flagged at this frequency. Move the time slider.",
        kind="warn"))
    _pairs, _vis = read_mean_visibility(
        _capture, _samples, _channel, pol_picker.value)
    _run = report_path.parent.parent
    _config = json.loads((_run / "starspin_config.json").read_text())
    _catalogue = json.loads((_run / "point_sources.json").read_text())
    _positions = {int(_id): np.array([_v["east"], _v["north"], _v["up"]])
                  for _fpga in _config["antenna_positions"].values()
                  for _id, _v in _fpga.items()}
    _ids = gain_data["antenna_ids"]
    _g = gain_data[_key]
    if gain_choice.value == "fine":
        _g = fine_data["gains"][_channel]
    elif gain_choice.value == "delay":
        _f0 = report["frequencies_hz"][_fi]
        _g = _g * np.exp(2j * np.pi * (mask_freq[_channel] - _f0) *
                         fine_data["delay_s"])
    mo.stop(not np.all(np.isfinite(_g)), mo.callout(
        "This fine bin has no finite candidate gain.", kind="warn"))
    _gains = dict(zip(_ids, _g))
    _held = gain_data["held_weights"]
    _id_index = {int(_id): _i for _i, _id in enumerate(_ids)}
    _allowed = np.array([
        (int(_a) in _id_index and int(_b) in _id_index and
         (baseline_picker.value != "held" or
          _held[_id_index[int(_a)], _id_index[int(_b)]] > 0))
        for _a, _b in _pairs], dtype=bool)
    _f = mask_freq[_channel]
    _grid, _raw, _n = dirty_image(_pairs, _vis, _positions, _f,
                                 allowed_pairs=_allowed)
    _, _cal, _ = dirty_image(_pairs, _vis, _positions, _f, gains=_gains,
                              allowed_pairs=_allowed)
    _sources = source_offsets(_catalogue, _config,
                              gain_data["times_mjd"][time_picker.value])
    _limits = _grid[[0, -1]]
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for _ax, _image, _title in zip(_axes, (_raw, _cal),
                                   ("Measured visibilities", "Candidate gain divided")):
        _center = np.nanmedian(_image)
        _display = _image - _center
        _scale = np.nanpercentile(np.abs(_display), 99)
        _ax.imshow(_display, origin="lower", cmap="coolwarm",
                   vmin=-_scale, vmax=_scale,
                   extent=[*_limits, *_limits])
        for _name, _east, _north in _sources:
            if abs(_east) < abs(_grid[-1]) and abs(_north) < abs(_grid[-1]):
                _ax.scatter(_east, _north, s=125, marker="o", facecolors="yellow",
                            edgecolors="black", linewidths=1.8, zorder=10)
                if show_source_labels.value:
                    _ax.annotate(_name, (_east, _north), xytext=(7, 8),
                             textcoords="offset points", color="yellow",
                             fontsize=11, fontweight="bold", zorder=11,
                             bbox={"facecolor": "black", "edgecolor": "none",
                                   "alpha": .8, "pad": 2})
        _ax.add_patch(Circle((0, 0), 1, fill=False, edgecolor="yellow",
                              linewidth=2, zorder=9))
        _ax.set(xlabel="East direction cosine (l)",
                ylabel="North direction cosine (m)", title=_title,
                xlim=(-1, 1), ylim=(-1, 1))
    _fig.suptitle(f"{_f / 1e6:.4f} MHz · {len(_samples)} clean integrations "
                  f"near {_integration} · {_n} cross baselines · {pol_picker.value} · "
                  f"{baseline_picker.value}")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Wideband image: all unmasked fine channels

    This combines every RFI-unmasked fine-channel cell in the selected short
    time window. Each channel is imaged at its own frequency, after its own
    candidate gain correction, then combined on a UV grid. The map keeps the
    full 512-pixel visible hemisphere; the view presets crop the same pixels
    so detail around Fornax A is retained. **Use the toolbar to pan and zoom**
    further, or hover to read the east/north direction cosines. The two panels
    are measured YY and gain-divided YY. Intensity is in arbitrary units; no
    CLEAN or primary-beam correction has been applied.
    """)
    return


@app.cell
def _(Path, mo):
    _default = Path("/data/BLACKMESA_1/sma075/cuda-spatial-filtering/analysis/"
                    "results/yy_wideband_30s.npz")
    wideband_path_text = mo.ui.text(value=str(_default), full_width=True,
                                    label="Cached wideband image path")
    wideband_path_text
    return wideband_path_text


@app.cell
def _(Path, mo, np, wideband_path_text):
    _path = Path(wideband_path_text.value).expanduser()
    mo.stop(not _path.is_file(), mo.callout(
        "Build the wideband cache with analysis/build_wideband_image.py or choose an existing .npz file.",
        kind="info"))
    with np.load(_path) as _z:
        wideband = {k: _z[k] for k in _z.files}
    return wideband


@app.cell
def _(mo):
    wide_target = mo.ui.dropdown(
        options=["Zenith", "For A", "Pic A", "Hyd A", "Tau A", "3C123"],
        value="For A", label="View centre")
    wide_scale = mo.ui.dropdown(
        options=["Full hemisphere", "60°", "30°", "15°", "7.5°", "3°"],
        value="Full hemisphere", label="Field of view")
    wide_gain = mo.ui.dropdown(
        options={"All-fine YY gains": 1, "Single-delay YY model": 2},
        value="All-fine YY gains", label="Right panel gain candidate")
    wide_baselines = mo.ui.dropdown(
        options={"All usable baselines": 0, "Held-out baselines": 1},
        value="All usable baselines", label="Baselines")
    mo.hstack([wide_target, wide_scale, wide_gain, wide_baselines],
              justify="start", gap=1)
    return wide_target, wide_scale, wide_gain, wide_baselines


@app.cell
def _(Circle, json, mo, np, plt, report_path, show_source_labels, source_offsets,
          wide_baselines, wide_gain, wide_scale, wide_target, wideband):
    _run = report_path.parent.parent
    _config = json.loads((_run / "starspin_config.json").read_text())
    _catalogue = json.loads((_run / "point_sources.json").read_text())
    _sources = source_offsets(_catalogue, _config,
                              float(np.median(wideband["time_mjd"])))
    _centres = {_name: (_east, _north) for _name, _east, _north in _sources}
    _centre = _centres.get(wide_target.value, (0.0, 0.0))
    _width = {"Full hemisphere": 2.0, "60°": 2 * np.sin(np.deg2rad(30)),
              "30°": 2 * np.sin(np.deg2rad(15)),
              "15°": 2 * np.sin(np.deg2rad(7.5)),
              "7.5°": 2 * np.sin(np.deg2rad(3.75)),
              "3°": 2 * np.sin(np.deg2rad(1.5))}[wide_scale.value]
    if wide_scale.value == "Full hemisphere":
        _centre = (0.0, 0.0)
    _axis = wideband["axis_lm"]
    _baseline = wide_baselines.value
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    for _ax, _mode, _title in ((_axes[0], 0, "Measured YY"),
                                (_axes[1], wide_gain.value, "Candidate gain divided YY")):
        _image = wideband["images"][_baseline, _mode]
        _display = _image - np.nanmedian(_image)
        _scale = np.nanpercentile(np.abs(_display), 99)
        _ax.imshow(_display, origin="lower", cmap="coolwarm",
                   vmin=-_scale, vmax=_scale,
                   extent=[_axis[0], _axis[-1], _axis[0], _axis[-1]],
                   interpolation="nearest")
        _ax.add_patch(Circle((0, 0), 1, fill=False,
                              edgecolor="yellow", linewidth=2, zorder=9))
        for _name, _east, _north in _sources:
            if (_east - _centre[0])**2 < _width**2 and (_north - _centre[1])**2 < _width**2:
                _ax.scatter(_east, _north, s=105, facecolors="yellow",
                            edgecolors="black", linewidths=1.6, zorder=10)
                if show_source_labels.value:
                    _ax.annotate(_name, (_east, _north), xytext=(6, 7),
                                 textcoords="offset points", color="yellow",
                                 fontsize=10, fontweight="bold", zorder=11,
                                 bbox={"facecolor": "black", "edgecolor": "none",
                                       "alpha": .8, "pad": 2})
        _ax.set(xlim=(_centre[0] - _width/2, _centre[0] + _width/2),
                ylim=(_centre[1] - _width/2, _centre[1] + _width/2),
                xlabel="East direction cosine (l)",
                ylabel="North direction cosine (m)",
                title=f"{_title} · {int(wideband['fine_bins'][_baseline, _mode])} bins")
    _fig.suptitle("All unmasked fine channels · short-time YY diagnostic")
    mo.mpl.interactive(_fig)
    return


if __name__ == "__main__":
    app.run()
