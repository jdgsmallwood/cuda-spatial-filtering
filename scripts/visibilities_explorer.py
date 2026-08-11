import marimo

__generated_with = "0.18.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # LAMBDA Visibilities Explorer

    Select a visibility HDF5 file to inspect its schema, run attributes,
    canonical receiver/antenna mapping, packed baseline order, zeroed inputs,
    missing-packet statistics, and visibility amplitude/phase.

    Launch with `marimo run scripts/visibilities_explorer.py`. If needed, install
    `marimo h5py pandas matplotlib` into the analysis environment.

    New files are self-describing. If a matching `*.streams.csv` sidecar is
    present, the raw FPGA datastream wiring is shown as well.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import json
    import h5py
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    return Path, h5py, json, mo, np, pd, plt


@app.cell
def _(Path, mo):
    dir_input = mo.ui.text(
        value=str(Path.cwd()),
        label="Directory",
        placeholder="Absolute path to search",
        full_width=True,
    )
    dir_input
    return (dir_input,)


@app.cell
def _(Path, dir_input, mo):
    _dir = Path(dir_input.value)
    _matches = (
        sorted(
            p for p in _dir.iterdir()
            if p.name.startswith("visibilities_") and p.suffix in (".h5", ".hdf5")
        )
        if _dir.is_dir()
        else []
    )
    file_picker = mo.ui.dropdown(
        options={p.name: p for p in _matches},
        value=_matches[0].name if _matches else None,
        label="Visibility file (visibilities_*.h5/.hdf5)",
        searchable=True,
    )
    file_picker
    return (file_picker,)


@app.cell
def _(file_picker, h5py, json, mo, np, pd):
    mo.stop(
        file_picker.value is None,
        mo.callout("Choose a visibility HDF5 file to begin.", kind="info"),
    )

    visibility_path = file_picker.value
    _inventory_rows = []
    _file_attributes = {}
    with h5py.File(visibility_path, "r") as _hdf:
        _file_attributes = {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in _hdf.attrs.items()
        }

        def _collect_item(name, obj):
            if isinstance(obj, h5py.Dataset):
                _inventory_rows.append(
                    {
                        "dataset": name,
                        "shape": str(obj.shape),
                        "dtype": str(obj.dtype),
                        "size": int(obj.size),
                        "compression": obj.compression or "none",
                    }
                )
            else:
                _inventory_rows.append(
                    {
                        "dataset": name + "/",
                        "shape": "group",
                        "dtype": "",
                        "size": "",
                        "compression": "",
                    }
                )

        _hdf.visititems(_collect_item)
        visibility_shape = (
            tuple(_hdf["visibilities"].shape)
            if "visibilities" in _hdf
            else None
        )
        antenna_ids = (
            _hdf["antenna_ids"][:].astype(int)
            if "antenna_ids" in _hdf
            else np.array([], dtype=int)
        )
        antenna_zeroed = (
            _hdf["antenna_zeroed"][:].astype(bool)
            if "antenna_zeroed" in _hdf
            else antenna_ids < 0
        )
        baseline_ids = (
            _hdf["baseline_ids"][:].astype(int)
            if "baseline_ids" in _hdf
            else np.array([], dtype=int)
        )
        baseline_receivers = (
            _hdf["baseline_receiver_indices"][:].astype(int)
            if "baseline_receiver_indices" in _hdf
            else np.empty((0, 2), dtype=int)
        )
        baseline_antennas = (
            _hdf["baseline_antenna_ids"][:].astype(int)
            if "baseline_antenna_ids" in _hdf
            else np.empty((0, 2), dtype=int)
        )
        baseline_zeroed = (
            _hdf["baseline_zeroed"][:].astype(bool)
            if "baseline_zeroed" in _hdf
            else np.array([], dtype=bool)
        )
        missing_stats = (
            _hdf["vis_missing_nums"][:]
            if "vis_missing_nums" in _hdf
            else np.empty((0, 3))
        )
        sequence_numbers = (
            _hdf["vis_seq_nums"][:]
            if "vis_seq_nums" in _hdf
            else np.empty((0, 2), dtype=int)
        )

        manifest = {}
        if "audit/run_manifest_json" in _hdf:
            _manifest_text = _hdf["audit/run_manifest_json"][()]
            if isinstance(_manifest_text, bytes):
                _manifest_text = _manifest_text.decode("utf-8")
            manifest = json.loads(str(_manifest_text))

        def _audit_table(dataset_name):
            if dataset_name not in _hdf:
                return pd.DataFrame()
            _dataset = _hdf[dataset_name]
            _columns = _dataset.attrs.get("columns", "")
            if isinstance(_columns, bytes):
                _columns = _columns.decode("utf-8")
            return pd.DataFrame(_dataset[:], columns=str(_columns).split(","))

        forward_mapping_df = _audit_table("audit/forward_stream_mapping")
        reverse_mapping_df = _audit_table("audit/reverse_canonical_mapping")
    if visibility_shape and baseline_receivers.size == 0:
        _baseline_count = visibility_shape[2]
        _pairs = [
            (receiver_1, receiver_2)
            for receiver_2 in range(
                int((np.sqrt(1 + 8 * _baseline_count) - 1) / 2)
            )
            for receiver_1 in range(receiver_2 + 1)
        ]
        baseline_receivers = np.asarray(_pairs, dtype=int)

    if baseline_antennas.size == 0 and baseline_ids.size:
        _invalid_id = int(_file_attributes.get("invalid_baseline_id", np.iinfo(np.int32).min))
        _valid_ids = baseline_ids != _invalid_id
        baseline_antennas = np.full((len(baseline_ids), 2), -1, dtype=int)
        baseline_antennas[_valid_ids, 0] = baseline_ids[_valid_ids] // 256
        baseline_antennas[_valid_ids, 1] = baseline_ids[_valid_ids] % 256
    if baseline_zeroed.size == 0 and baseline_antennas.size:
        baseline_zeroed = np.any(baseline_antennas < 0, axis=1)

    inventory_df = pd.DataFrame(_inventory_rows)
    attributes_df = pd.DataFrame(
        [{"attribute": key, "value": value} for key, value in _file_attributes.items()]
    )
    sidecar_path = visibility_path.with_suffix(".streams.csv")
    stream_df = (
        pd.read_csv(sidecar_path)
        if sidecar_path.exists()
        else pd.DataFrame()
    )

    return (
        antenna_ids,
        antenna_zeroed,
        attributes_df,
        baseline_antennas,
        baseline_ids,
        baseline_receivers,
        baseline_zeroed,
        inventory_df,
        missing_stats,
        manifest,
        forward_mapping_df,
        reverse_mapping_df,
        sequence_numbers,
        sidecar_path,
        stream_df,
        visibility_path,
        visibility_shape,
    )


@app.cell(hide_code=True)
def _(forward_mapping_df, manifest, mo, pd, reverse_mapping_df):
    if manifest:
        _build_df = pd.DataFrame(
            [{"field": key, "value": value} for key, value in manifest.get("build", {}).items()]
        )
        _environment_df = pd.DataFrame(
            [{"variable": key, "value": value} for key, value in manifest.get("environment", {}).items()]
        )
        _input_files_df = pd.DataFrame(
            [
                {
                    "input": name,
                    "path": details.get("path", ""),
                    "embedded": details.get("present", False),
                    "bytes": len(details.get("content", "")),
                }
                for name, details in manifest.get("input_files", {}).items()
            ]
        )
        _manifest_view = mo.vstack(
            [
                mo.md(
                    "## Embedded run configuration\n\n"
                    f"**Command:** `{' '.join(manifest.get('command_line', []))}`"
                ),
                mo.md("### Build provenance"),
                mo.ui.table(_build_df, selection=None),
                mo.md("### Runtime environment"),
                mo.ui.table(_environment_df, selection=None),
                mo.md("### Embedded input files"),
                mo.ui.table(_input_files_df, selection=None, pagination=True),
            ]
        )
    else:
        _manifest_view = mo.callout(
            "This older file has no embedded run manifest.", kind="warn"
        )

    _mapping_view = (
        mo.vstack(
            [
                mo.md("## Bidirectional mapping tables"),
                mo.md("### Raw datastream → canonical antenna/polarization"),
                mo.ui.table(forward_mapping_df, selection=None, pagination=True),
                mo.md("### Canonical antenna/polarization → raw datastream"),
                mo.ui.table(reverse_mapping_df, selection=None, pagination=True),
            ]
        )
        if not forward_mapping_df.empty
        else mo.callout(
            "No embedded bidirectional mapping tables are present.", kind="info"
        )
    )
    mo.vstack([_manifest_view, _mapping_view])
    return


@app.cell(hide_code=True)
def _(attributes_df, inventory_df, mo, visibility_path, visibility_shape):
    mo.vstack(
        [
            mo.md(
                f"## File overview\n\n"
                f"**Path:** `{visibility_path}`  \n"
                f"**Visibility shape:** `{visibility_shape}`  \n"
                "Axes: `integration, channel, baseline, pol-1, pol-2, real/imag`."
            ),
            mo.md("### File attributes"),
            mo.ui.table(attributes_df, selection=None),
            mo.md("### Dataset inventory"),
            mo.ui.table(inventory_df, selection=None, pagination=True),
        ]
    )
    return


@app.cell
def _(
    antenna_ids,
    antenna_zeroed,
    baseline_antennas,
    baseline_ids,
    baseline_receivers,
    baseline_zeroed,
    np,
    pd,
):
    _receiver_count = (
        len(antenna_ids)
        if len(antenna_ids)
        else (int(baseline_receivers.max()) + 1 if baseline_receivers.size else 0)
    )
    receiver_df = pd.DataFrame(
        {
            "canonical_receiver_index": np.arange(_receiver_count),
            "antenna_id": (
                antenna_ids
                if len(antenna_ids)
                else np.full(_receiver_count, -1, dtype=int)
            ),
            "zeroed": (
                antenna_zeroed
                if len(antenna_zeroed)
                else np.full(_receiver_count, False)
            ),
        }
    )
    _baseline_count = len(baseline_receivers)
    baseline_df = pd.DataFrame(
        {
            "baseline_index": np.arange(_baseline_count),
            "receiver_1": baseline_receivers[:, 0] if _baseline_count else [],
            "receiver_2": baseline_receivers[:, 1] if _baseline_count else [],
            "antenna_1": (
                baseline_antennas[:, 0]
                if len(baseline_antennas)
                else np.full(_baseline_count, -1)
            ),
            "antenna_2": (
                baseline_antennas[:, 1]
                if len(baseline_antennas)
                else np.full(_baseline_count, -1)
            ),
            "packed_baseline_id": (
                baseline_ids
                if len(baseline_ids)
                else np.full(_baseline_count, -1)
            ),
            "zeroed": (
                baseline_zeroed
                if len(baseline_zeroed)
                else np.full(_baseline_count, False)
            ),
        }
    )
    return baseline_df, receiver_df


@app.cell(hide_code=True)
def _(baseline_df, mo, receiver_df, sidecar_path, stream_df):
    _stream_view = (
        mo.ui.table(stream_df, selection=None, pagination=True)
        if not stream_df.empty
        else mo.callout(
            f"No stream audit sidecar found at `{sidecar_path}`. "
            "The HDF5 canonical metadata is still available.",
            kind="warn",
        )
    )
    mo.vstack(
        [
            mo.md("## Antenna and baseline audit"),
            mo.md("### Canonical receivers"),
            mo.ui.table(receiver_df, selection=None, pagination=True),
            mo.md("### Packed baseline axis"),
            mo.ui.table(baseline_df, selection=None, pagination=True),
            mo.md("### Raw datastream mapping"),
            _stream_view,
        ]
    )
    return


@app.cell(hide_code=True)
def _(forward_mapping_df, mo, np, plt):
    mo.stop(
        forward_mapping_df.empty,
        mo.callout(
            "No embedded `audit/forward_stream_mapping` in this file — wiring grid unavailable.",
            kind="info",
        ),
    )
    # One row per (FPGA, receiver slot): filter to raw_polarization == 0
    _df = forward_mapping_df[forward_mapping_df["raw_polarization"] == 0].copy()
    _fpga_ids = sorted(_df["fpga_id"].unique())
    _n_fpgas = len(_fpga_ids)
    _n_slots = int(_df["receiver_slot"].max()) + 1

    _ant_id = np.full((_n_fpgas, _n_slots), -1, dtype=int)
    _canon = np.full((_n_fpgas, _n_slots), -1, dtype=int)
    _disconnected = np.ones((_n_fpgas, _n_slots), dtype=bool)
    for _row in _df.itertuples():
        _fi = _fpga_ids.index(int(_row.fpga_id))
        _s = int(_row.receiver_slot)
        _ant_id[_fi, _s] = int(_row.antenna_id)
        _canon[_fi, _s] = int(_row.canonical_receiver_index)
        _disconnected[_fi, _s] = bool(_row.configured_disconnected)

    _color = np.ma.masked_where(_disconnected, _canon.astype(float))
    _cmap = plt.cm.tab20.copy()
    _cmap.set_bad(color="#d0d0d0")

    _fig, _ax = plt.subplots(
        figsize=(max(8, _n_slots * 1.1), _n_fpgas * 1.6 + 0.8),
        constrained_layout=True,
    )
    _im = _ax.imshow(_color, aspect="auto", interpolation="nearest", cmap=_cmap,
                     vmin=0, vmax=max(1, int(_color.max()) if not _color.mask.all() else 1))
    for _fi in range(_n_fpgas):
        for _s in range(_n_slots):
            if _disconnected[_fi, _s]:
                _ax.text(_s, _fi, "—", ha="center", va="center", fontsize=9, color="#999")
            else:
                _ax.text(_s, _fi - 0.15, f"A{_ant_id[_fi, _s]}",
                         ha="center", va="center", fontsize=9, fontweight="bold", color="white")
                _ax.text(_s, _fi + 0.25, f"R{_canon[_fi, _s]}",
                         ha="center", va="center", fontsize=7, color="white", alpha=0.85)

    _ax.set_xticks(np.arange(_n_slots))
    _ax.set_xticklabels([f"slot {_s}" for _s in range(_n_slots)])
    _ax.set_yticks(np.arange(_n_fpgas))
    _ax.set_yticklabels([f"FPGA {_f}" for _f in _fpga_ids])
    _ax.set_xlabel("Receiver slot within FPGA")
    _ax.set_title("Stream wiring — A = antenna_id, R = canonical receiver index")
    _fig.colorbar(_im, ax=_ax, label="Canonical receiver index")
    _ax.set_xlim(-0.5, _n_slots - 0.5)
    _ax.set_ylim(_n_fpgas - 0.5, -0.5)
    mo.vstack([mo.md("## Wiring grid"), _fig])
    return


@app.cell
def _(Path, mo, visibility_path):
    _default = str(visibility_path.parent / "config.json")
    layout_config_input = mo.ui.text(
        value=_default,
        label="config.json for physical layout (optional)",
        full_width=True,
    )
    layout_config_input
    return (layout_config_input,)


@app.cell(hide_code=True)
def _(Path, forward_mapping_df, json, layout_config_input, mo, np, plt, receiver_df):
    _cfg_path = Path(layout_config_input.value)
    mo.stop(
        not _cfg_path.exists(),
        mo.callout(
            f"`{_cfg_path}` not found — physical layout unavailable. "
            "Point the input above at your config.json.",
            kind="info",
        ),
    )
    with open(_cfg_path) as _f:
        _cfg = json.load(_f)

    # {antenna_id: (east_m, north_m)}
    _enu = {}
    for _fpga_antennas in _cfg.get("antenna_positions", {}).values():
        for _ant_str, _pos in _fpga_antennas.items():
            _enu[int(_ant_str)] = (_pos["east"], _pos["north"])

    # antenna_id → FPGA (from forward mapping if available)
    _ant_to_fpga = {}
    if not forward_mapping_df.empty:
        for _row in forward_mapping_df[forward_mapping_df["raw_polarization"] == 0].itertuples():
            if int(_row.antenna_id) >= 0:
                _ant_to_fpga[int(_row.antenna_id)] = int(_row.fpga_id)

    _fpga_colors = {0: "tab:blue", 1: "tab:orange", 2: "tab:green", 3: "tab:red"}
    _fig, _ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    _ax.set_aspect("equal")
    _ax.axhline(0, color="#ddd", linewidth=0.8)
    _ax.axvline(0, color="#ddd", linewidth=0.8)
    _ax.scatter([0], [0], marker="+", s=120, color="black", zorder=5)
    _ax.text(0.15, 0.15, "ref", fontsize=7, color="black")

    _seen_fpgas = set()
    for _row in receiver_df.itertuples():
        _aid = int(_row.antenna_id)
        if _aid < 0 or _aid not in _enu:
            continue
        _e, _n = _enu[_aid]
        _fpga = _ant_to_fpga.get(_aid, -1)
        _color = _fpga_colors.get(_fpga, "gray")
        _lbl = f"FPGA {_fpga}" if _fpga >= 0 else "unknown"
        _ax.scatter([_e], [_n], color=_color, s=70, zorder=4,
                    label=None if _fpga in _seen_fpgas else _lbl)
        _seen_fpgas.add(_fpga)
        _canon = int(_row.canonical_receiver_index)
        _ax.annotate(
            f"A{_aid}\nR{_canon}",
            xy=(_e, _n),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
            color=_color,
            ha="left",
        )

    _ax.set_xlabel("East (m)")
    _ax.set_ylabel("North (m)")
    _ax.set_title("Physical antenna layout (ENU)  —  A = antenna_id, R = canonical receiver index")
    _ax.legend(title="FPGA", fontsize=8)
    _ax.grid(True, alpha=0.25)
    mo.vstack([mo.md("## Physical antenna layout"), _fig])
    return


@app.cell
def _(baseline_df, mo, visibility_shape):
    mo.stop(
        visibility_shape is None,
        mo.callout("This file has no `visibilities` dataset.", kind="danger"),
    )
    _baseline_options = {
        (
            f"{int(row.baseline_index)}: "
            f"R{int(row.receiver_1)}–R{int(row.receiver_2)} / "
            f"A{int(row.antenna_1)}–A{int(row.antenna_2)}"
            + (" [ZEROED]" if bool(row.zeroed) else "")
        ): int(row.baseline_index)
        for row in baseline_df.itertuples()
    }
    baseline_picker = mo.ui.dropdown(
        options=_baseline_options,
        value=next(iter(_baseline_options)) if _baseline_options else None,
        label="Baseline",
        searchable=True,
    )
    pol_1_picker = mo.ui.dropdown(
        options={"X (0)": 0, "Y (1)": 1}, value="X (0)", label="Polarization 1"
    )
    pol_2_picker = mo.ui.dropdown(
        options={"X (0)": 0, "Y (1)": 1}, value="X (0)", label="Polarization 2"
    )
    _controls = mo.hstack(
        [baseline_picker, pol_1_picker, pol_2_picker],
        justify="start",
    )
    _controls
    return baseline_picker, pol_1_picker, pol_2_picker


@app.cell
def _(
    baseline_picker,
    h5py,
    np,
    pol_1_picker,
    pol_2_picker,
    visibility_path,
):
    with h5py.File(visibility_path, "r") as _hdf:
        _raw_visibility = _hdf["visibilities"][
            :,
            :,
            baseline_picker.value,
            pol_1_picker.value,
            pol_2_picker.value,
            :,
        ]
    selected_visibility = _raw_visibility[..., 0] + 1j * _raw_visibility[..., 1]
    selected_amplitude = np.abs(selected_visibility)
    selected_phase = np.angle(selected_visibility)
    return selected_amplitude, selected_phase


@app.cell(hide_code=True)
def _(
    baseline_picker,
    mo,
    np,
    plt,
    pol_1_picker,
    pol_2_picker,
    selected_amplitude,
    selected_phase,
):
    _figure, _axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    _amplitude_image = _axes[0].imshow(
        selected_amplitude,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )
    _axes[0].set_ylabel("Integration")
    _axes[0].set_title("Visibility amplitude")
    _figure.colorbar(_amplitude_image, ax=_axes[0], label="Amplitude")

    _phase_image = _axes[1].imshow(
        selected_phase,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        vmin=-np.pi,
        vmax=np.pi,
    )
    _axes[1].set_xlabel("Channel index")
    _axes[1].set_ylabel("Integration")
    _axes[1].set_title("Visibility phase")
    _figure.colorbar(_phase_image, ax=_axes[1], label="Radians")

    mo.vstack(
        [
            mo.md(
                f"## Selected visibility\n\n"
                f"Baseline **{baseline_picker.value}**, "
                f"polarizations **{pol_1_picker.value} × {pol_2_picker.value}**"
            ),
            _figure,
        ]
    )
    return


@app.cell
def _(mo, visibility_shape):
    mo.stop(
        visibility_shape is None,
        mo.callout("No `visibilities` dataset for correlation matrix.", kind="danger"),
    )
    _n_integrations, _n_channels = visibility_shape[0], visibility_shape[1]
    corr_channel_picker = mo.ui.slider(
        start=0,
        stop=_n_channels - 1,
        value=_n_channels // 2,
        label="Channel (correlation matrix)",
    )
    corr_integration_picker = mo.ui.slider(
        start=0,
        stop=_n_integrations - 1,
        value=0,
        label="Integration (correlation matrix)",
    )
    corr_display_picker = mo.ui.radio(
        options=["Amplitude", "Phase"],
        value="Amplitude",
        label="Display",
    )
    corr_log_picker = mo.ui.checkbox(label="Log scale (amplitude only)", value=False)
    mo.hstack(
        [corr_channel_picker, corr_integration_picker, corr_display_picker, corr_log_picker],
        justify="start",
    )
    return corr_channel_picker, corr_display_picker, corr_integration_picker, corr_log_picker


@app.cell
def _(
    baseline_receivers,
    corr_channel_picker,
    corr_integration_picker,
    h5py,
    np,
    visibility_path,
):
    mo_stop_corr = baseline_receivers.size == 0
    if not mo_stop_corr:
        with h5py.File(visibility_path, "r") as _hdf:
            _raw = _hdf["visibilities"][
                corr_integration_picker.value,
                corr_channel_picker.value,
                :, :, :, :,
            ]
        _vis = _raw[..., 0] + 1j * _raw[..., 1]  # (n_baselines, 2, 2)
        _n_recv = int(baseline_receivers.max()) + 1
        corr_matrix = np.zeros((_n_recv, _n_recv, 2, 2), dtype=complex)
        _r1 = baseline_receivers[:, 0]
        _r2 = baseline_receivers[:, 1]
        corr_matrix[_r1, _r2] = _vis
        _off = _r1 != _r2
        corr_matrix[_r2[_off], _r1[_off]] = np.conj(_vis[_off].swapaxes(-2, -1))
    else:
        corr_matrix = np.zeros((0, 0, 2, 2), dtype=complex)
    return corr_matrix, mo_stop_corr


@app.cell(hide_code=True)
def _(
    antenna_ids,
    corr_channel_picker,
    corr_display_picker,
    corr_integration_picker,
    corr_log_picker,
    corr_matrix,
    mo,
    mo_stop_corr,
    np,
    plt,
):
    mo.stop(mo_stop_corr, mo.callout("No baseline data available for correlation matrix.", kind="warn"))
    _n = corr_matrix.shape[0]
    _ant_labels = (
        [str(_aid) for _aid in antenna_ids]
        if len(antenna_ids) == _n
        else [str(_i) for _i in range(_n)]
    )
    _axis_label = "Antenna ID" if len(antenna_ids) == _n else "Receiver index"

    # Tile all 4 pol combinations into a single 2N×2N matrix:
    #   top-left=XX, top-right=XY, bottom-left=YX, bottom-right=YY
    _tiled = np.zeros((2 * _n, 2 * _n))
    for (_p1, _p2), (_qr, _qc) in zip(
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
    ):
        _block = corr_matrix[:, :, _p1, _p2]
        _tiled[_qr * _n : (_qr + 1) * _n, _qc * _n : (_qc + 1) * _n] = (
            np.abs(_block) if corr_display_picker.value == "Amplitude" else np.angle(_block)
        )

    _im_kwargs = {}
    if corr_display_picker.value == "Amplitude":
        if corr_log_picker.value:
            _eps = np.finfo(float).tiny
            _tiled = np.log10(np.maximum(_tiled, _eps))
            _cb_label = "log₁₀(Amplitude)"
        else:
            _cb_label = "Amplitude"
    else:
        _cb_label = "Phase (rad)"
        _im_kwargs = {"vmin": -np.pi, "vmax": np.pi, "cmap": "hsv"}

    _figure, _ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
    _im = _ax.imshow(
        _tiled,
        aspect="equal",
        origin="upper",
        interpolation="nearest",
        **_im_kwargs,
    )
    _figure.colorbar(_im, ax=_ax, label=_cb_label)

    # Quadrant dividers
    _ax.axhline(_n - 0.5, color="white", linewidth=1.5, linestyle="--", alpha=0.7)
    _ax.axvline(_n - 0.5, color="white", linewidth=1.5, linestyle="--", alpha=0.7)

    # Quadrant labels
    for _qlabel, _qcy, _qcx in [
        ("XX", _n / 2 - 0.5, _n / 2 - 0.5),
        ("XY", _n / 2 - 0.5, 3 * _n / 2 - 0.5),
        ("YX", 3 * _n / 2 - 0.5, _n / 2 - 0.5),
        ("YY", 3 * _n / 2 - 0.5, 3 * _n / 2 - 0.5),
    ]:
        _ax.text(
            _qcx, _qcy, _qlabel,
            ha="center", va="center",
            fontsize=14, fontweight="bold",
            color="white", alpha=0.55,
        )

    # Ticks: N per pol block, labeled by antenna ID repeated for X then Y
    _ticks = np.arange(2 * _n)
    _tick_labels = [f"X:{_l}" for _l in _ant_labels] + [f"Y:{_l}" for _l in _ant_labels]
    _ax.set_xticks(_ticks)
    _ax.set_yticks(_ticks)
    _ax.set_xticklabels(_tick_labels, rotation=90, fontsize=6)
    _ax.set_yticklabels(_tick_labels, fontsize=6)
    _ax.set_xlabel(_axis_label)
    _ax.set_ylabel(_axis_label)
    _ax.set_title(
        f"Correlation matrix — channel {corr_channel_picker.value}, "
        f"integration {corr_integration_picker.value}"
    )
    mo.vstack([mo.md("## Correlation matrix"), _figure])
    return


@app.cell(hide_code=True)
def _(missing_stats, mo, pd, sequence_numbers):
    if len(missing_stats):
        _packet_df = pd.DataFrame(
            {
                "integration": range(len(missing_stats)),
                "start_sequence": sequence_numbers[:, 0],
                "end_sequence": sequence_numbers[:, 1],
                "missing_packets": missing_stats[:, 0],
                "total_packets": missing_stats[:, 1],
                "missing_percent": missing_stats[:, 2],
            }
        )
        _packet_view = mo.ui.table(_packet_df, selection=None, pagination=True)
    else:
        _packet_view = mo.callout("No packet-loss dataset in this file.", kind="info")
    mo.vstack([mo.md("## Integration and packet-loss audit"), _packet_view])
    return


@app.cell
def _(inventory_df, mo):
    _dataset_names = [
        name for name in inventory_df["dataset"].tolist() if not name.endswith("/")
    ]
    dataset_picker = mo.ui.dropdown(
        options=_dataset_names,
        value=_dataset_names[0] if _dataset_names else None,
        label="Dataset preview",
        searchable=True,
    )
    dataset_picker
    return (dataset_picker,)


@app.cell(hide_code=True)
def _(dataset_picker, h5py, mo, np, pd, visibility_path):
    with h5py.File(visibility_path, "r") as _hdf:
        _dataset = _hdf[dataset_picker.value]
        _selection = tuple(slice(0, min(2, size)) for size in _dataset.shape)
        _preview_array = np.asarray(_dataset[_selection] if _selection else _dataset[()])
        _preview_flat = _preview_array.reshape(-1)[:200]
    _preview_df = pd.DataFrame(
        {"flat_preview_index": range(len(_preview_flat)), "value": _preview_flat}
    )
    mo.vstack(
        [
            mo.md(
                f"### Preview: `{dataset_picker.value}`\n\n"
                "At most two elements per axis and 200 flattened values are shown."
            ),
            mo.ui.table(_preview_df, selection=None, pagination=True),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
