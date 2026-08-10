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

    import h5py
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    return Path, h5py, mo, np, pd, plt


@app.cell
def _(Path, mo):
    file_browser = mo.ui.file_browser(
        initial_path=Path.cwd(),
        filetypes=[".h5", ".hdf5"],
        multiple=False,
        label="Visibility HDF5 file",
    )
    file_browser
    return (file_browser,)


@app.cell
def _(Path, file_browser, h5py, mo, np, pd):
    mo.stop(
        not file_browser.value,
        mo.callout("Choose a visibility HDF5 file to begin.", kind="info"),
    )

    visibility_path = Path(file_browser.path(index=0))
    inventory_rows = []
    file_attributes = {}
    with h5py.File(visibility_path, "r") as hdf:
        file_attributes = {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in hdf.attrs.items()
        }

        def collect_item(name, obj):
            if isinstance(obj, h5py.Dataset):
                inventory_rows.append(
                    {
                        "dataset": name,
                        "shape": str(obj.shape),
                        "dtype": str(obj.dtype),
                        "size": int(obj.size),
                        "compression": obj.compression or "none",
                    }
                )
            else:
                inventory_rows.append(
                    {
                        "dataset": name + "/",
                        "shape": "group",
                        "dtype": "",
                        "size": "",
                        "compression": "",
                    }
                )

        hdf.visititems(collect_item)
        visibility_shape = (
            tuple(hdf["visibilities"].shape)
            if "visibilities" in hdf
            else None
        )
        antenna_ids = (
            hdf["antenna_ids"][:].astype(int)
            if "antenna_ids" in hdf
            else np.array([], dtype=int)
        )
        antenna_zeroed = (
            hdf["antenna_zeroed"][:].astype(bool)
            if "antenna_zeroed" in hdf
            else antenna_ids < 0
        )
        baseline_ids = (
            hdf["baseline_ids"][:].astype(int)
            if "baseline_ids" in hdf
            else np.array([], dtype=int)
        )
        baseline_receivers = (
            hdf["baseline_receiver_indices"][:].astype(int)
            if "baseline_receiver_indices" in hdf
            else np.empty((0, 2), dtype=int)
        )
        baseline_antennas = (
            hdf["baseline_antenna_ids"][:].astype(int)
            if "baseline_antenna_ids" in hdf
            else np.empty((0, 2), dtype=int)
        )
        baseline_zeroed = (
            hdf["baseline_zeroed"][:].astype(bool)
            if "baseline_zeroed" in hdf
            else np.array([], dtype=bool)
        )
        missing_stats = (
            hdf["vis_missing_nums"][:]
            if "vis_missing_nums" in hdf
            else np.empty((0, 3))
        )
        sequence_numbers = (
            hdf["vis_seq_nums"][:]
            if "vis_seq_nums" in hdf
            else np.empty((0, 2), dtype=int)
        )

        manifest = {}
        if "audit/run_manifest_json" in hdf:
            manifest_text = hdf["audit/run_manifest_json"][()]
            if isinstance(manifest_text, bytes):
                manifest_text = manifest_text.decode("utf-8")
            manifest = json.loads(str(manifest_text))

        def audit_table(dataset_name):
            if dataset_name not in hdf:
                return pd.DataFrame()
            dataset = hdf[dataset_name]
            columns = dataset.attrs.get("columns", "")
            if isinstance(columns, bytes):
                columns = columns.decode("utf-8")
            return pd.DataFrame(dataset[:], columns=str(columns).split(","))

        forward_mapping_df = audit_table("audit/forward_stream_mapping")
        reverse_mapping_df = audit_table("audit/reverse_canonical_mapping")
    if visibility_shape and baseline_receivers.size == 0:
        baseline_count = visibility_shape[2]
        pairs = [
            (receiver_1, receiver_2)
            for receiver_2 in range(
                int((np.sqrt(1 + 8 * baseline_count) - 1) / 2)
            )
            for receiver_1 in range(receiver_2 + 1)
        ]
        baseline_receivers = np.asarray(pairs, dtype=int)

    if baseline_antennas.size == 0 and baseline_ids.size:
        invalid_id = int(file_attributes.get("invalid_baseline_id", np.iinfo(np.int32).min))
        valid_ids = baseline_ids != invalid_id
        baseline_antennas = np.full((len(baseline_ids), 2), -1, dtype=int)
        baseline_antennas[valid_ids, 0] = baseline_ids[valid_ids] // 256
        baseline_antennas[valid_ids, 1] = baseline_ids[valid_ids] % 256
    if baseline_zeroed.size == 0 and baseline_antennas.size:
        baseline_zeroed = np.any(baseline_antennas < 0, axis=1)

    inventory_df = pd.DataFrame(inventory_rows)
    attributes_df = pd.DataFrame(
        [{"attribute": key, "value": value} for key, value in file_attributes.items()]
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
        build_df = pd.DataFrame(
            [{"field": key, "value": value} for key, value in manifest.get("build", {}).items()]
        )
        environment_df = pd.DataFrame(
            [{"variable": key, "value": value} for key, value in manifest.get("environment", {}).items()]
        )
        input_files_df = pd.DataFrame(
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
        manifest_view = mo.vstack(
            [
                mo.md(
                    "## Embedded run configuration\n\n"
                    f"**Command:** `{' '.join(manifest.get('command_line', []))}`"
                ),
                mo.md("### Build provenance"),
                mo.ui.table(build_df, selection=None),
                mo.md("### Runtime environment"),
                mo.ui.table(environment_df, selection=None),
                mo.md("### Embedded input files"),
                mo.ui.table(input_files_df, selection=None, pagination=True),
            ]
        )
    else:
        manifest_view = mo.callout(
            "This older file has no embedded run manifest.", kind="warn"
        )

    mapping_view = (
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
    mo.vstack([manifest_view, mapping_view])
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
    receiver_count = (
        len(antenna_ids)
        if len(antenna_ids)
        else (int(baseline_receivers.max()) + 1 if baseline_receivers.size else 0)
    )
    receiver_df = pd.DataFrame(
        {
            "canonical_receiver_index": np.arange(receiver_count),
            "antenna_id": (
                antenna_ids
                if len(antenna_ids)
                else np.full(receiver_count, -1, dtype=int)
            ),
            "zeroed": (
                antenna_zeroed
                if len(antenna_zeroed)
                else np.full(receiver_count, False)
            ),
        }
    )
    baseline_count = len(baseline_receivers)
    baseline_df = pd.DataFrame(
        {
            "baseline_index": np.arange(baseline_count),
            "receiver_1": baseline_receivers[:, 0] if baseline_count else [],
            "receiver_2": baseline_receivers[:, 1] if baseline_count else [],
            "antenna_1": (
                baseline_antennas[:, 0]
                if len(baseline_antennas)
                else np.full(baseline_count, -1)
            ),
            "antenna_2": (
                baseline_antennas[:, 1]
                if len(baseline_antennas)
                else np.full(baseline_count, -1)
            ),
            "packed_baseline_id": (
                baseline_ids
                if len(baseline_ids)
                else np.full(baseline_count, -1)
            ),
            "zeroed": (
                baseline_zeroed
                if len(baseline_zeroed)
                else np.full(baseline_count, False)
            ),
        }
    )
    return baseline_df, receiver_df


@app.cell(hide_code=True)
def _(baseline_df, mo, receiver_df, sidecar_path, stream_df):
    stream_view = (
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
            stream_view,
        ]
    )
    return


@app.cell
def _(baseline_df, mo, visibility_shape):
    mo.stop(
        visibility_shape is None,
        mo.callout("This file has no `visibilities` dataset.", kind="danger"),
    )
    baseline_options = {
        (
            f"{int(row.baseline_index)}: "
            f"R{int(row.receiver_1)}–R{int(row.receiver_2)} / "
            f"A{int(row.antenna_1)}–A{int(row.antenna_2)}"
            + (" [ZEROED]" if bool(row.zeroed) else "")
        ): int(row.baseline_index)
        for row in baseline_df.itertuples()
    }
    baseline_picker = mo.ui.dropdown(
        options=baseline_options,
        value=next(iter(baseline_options)) if baseline_options else None,
        label="Baseline",
        searchable=True,
    )
    pol_1_picker = mo.ui.dropdown(
        options={"X (0)": 0, "Y (1)": 1}, value="X (0)", label="Polarization 1"
    )
    pol_2_picker = mo.ui.dropdown(
        options={"X (0)": 0, "Y (1)": 1}, value="X (0)", label="Polarization 2"
    )
    controls = mo.hstack(
        [baseline_picker, pol_1_picker, pol_2_picker],
        justify="start",
    )
    controls
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
    with h5py.File(visibility_path, "r") as hdf:
        raw_visibility = hdf["visibilities"][
            :,
            :,
            baseline_picker.value,
            pol_1_picker.value,
            pol_2_picker.value,
            :,
        ]
    selected_visibility = raw_visibility[..., 0] + 1j * raw_visibility[..., 1]
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
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    amplitude_image = axes[0].imshow(
        selected_amplitude,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )
    axes[0].set_ylabel("Integration")
    axes[0].set_title("Visibility amplitude")
    figure.colorbar(amplitude_image, ax=axes[0], label="Amplitude")

    phase_image = axes[1].imshow(
        selected_phase,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        vmin=-np.pi,
        vmax=np.pi,
    )
    axes[1].set_xlabel("Channel index")
    axes[1].set_ylabel("Integration")
    axes[1].set_title("Visibility phase")
    figure.colorbar(phase_image, ax=axes[1], label="Radians")

    mo.vstack(
        [
            mo.md(
                f"## Selected visibility\n\n"
                f"Baseline **{baseline_picker.value}**, "
                f"polarizations **{pol_1_picker.value} × {pol_2_picker.value}**"
            ),
            figure,
        ]
    )
    return


@app.cell(hide_code=True)
def _(missing_stats, mo, pd, sequence_numbers):
    if len(missing_stats):
        packet_df = pd.DataFrame(
            {
                "integration": range(len(missing_stats)),
                "start_sequence": sequence_numbers[:, 0],
                "end_sequence": sequence_numbers[:, 1],
                "missing_packets": missing_stats[:, 0],
                "total_packets": missing_stats[:, 1],
                "missing_percent": missing_stats[:, 2],
            }
        )
        packet_view = mo.ui.table(packet_df, selection=None, pagination=True)
    else:
        packet_view = mo.callout("No packet-loss dataset in this file.", kind="info")
    mo.vstack([mo.md("## Integration and packet-loss audit"), packet_view])
    return


@app.cell
def _(inventory_df, mo):
    dataset_names = [
        name for name in inventory_df["dataset"].tolist() if not name.endswith("/")
    ]
    dataset_picker = mo.ui.dropdown(
        options=dataset_names,
        value=dataset_names[0] if dataset_names else None,
        label="Dataset preview",
        searchable=True,
    )
    dataset_picker
    return (dataset_picker,)


@app.cell(hide_code=True)
def _(dataset_picker, h5py, mo, np, pd, visibility_path):
    with h5py.File(visibility_path, "r") as hdf:
        dataset = hdf[dataset_picker.value]
        selection = tuple(slice(0, min(2, size)) for size in dataset.shape)
        preview_array = np.asarray(dataset[selection] if selection else dataset[()])
        preview_flat = preview_array.reshape(-1)[:200]
    preview_df = pd.DataFrame(
        {"flat_preview_index": range(len(preview_flat)), "value": preview_flat}
    )
    mo.vstack(
        [
            mo.md(
                f"### Preview: `{dataset_picker.value}`\n\n"
                "At most two elements per axis and 200 flattened values are shown."
            ),
            mo.ui.table(preview_df, selection=None, pagination=True),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
