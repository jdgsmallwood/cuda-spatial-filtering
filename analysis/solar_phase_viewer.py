"""Current Sun calibration and imaging evidence.

Run with: marimo run analysis/solar_phase_viewer.py
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    return Path, json, mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Sun calibration: current results

    The images are in relative correlator units. Source labels show expected
    positions, not detections. The phase-drift interpolation and time-frequency
    mask were fitted retrospectively to this capture; the extra antenna
    amplitude and phase terms use only the previous 10-second interval.
    """)
    return


@app.cell
def _(Path, mo):
    _root = Path("/data/BLACKMESA_1/sma075/cuda-spatial-filtering/analysis/results")
    _default = _root / "sun_phase_high_20261003_1050/report.json"
    report_browser = mo.ui.file_browser(
        initial_path=_root, filetypes=[".json"], multiple=False,
        restrict_navigation=False, label="Choose a Sun capture report")
    report_text = mo.ui.text(value=str(_default), full_width=True,
                             label="Or paste a report.json path")
    mo.vstack([report_browser, report_text])
    return report_browser, report_text


@app.cell
def _(Path, json, mo, report_browser, report_text):
    _picked = report_browser.path(index=0) if report_browser.value else report_text.value
    report_path = Path(_picked).expanduser()
    mo.stop(not report_path.is_file(),
            mo.callout("Choose an existing Sun report.json.", kind="warn"))
    report = json.loads(report_path.read_text())
    mo.stop("summary" not in report or "sun_altitude_deg" not in report,
            mo.callout("This is not a Sun phase report.", kind="warn"))
    return report, report_path


@app.cell(hide_code=True)
def _(mo, report):
    _loss = report.get("packet_loss_fraction")
    _loss_text = f"{100 * _loss:.3f}%" if _loss is not None else "not recorded"
    _alt = report["sun_altitude_deg"]
    mo.md(f"""
    **Capture:** `{report['capture']}`  
    **Duration:** {report['duration_s'] / 60:.1f} min ·
    **Sun elevation:** {_alt[0]:.1f}° → {_alt[1]:.1f}° ·
    **Complete integrations:** {report['complete_integrations']} /
    {report['total_integrations']} · **Missing packets:** {_loss_text}
    """)
    return


@app.cell(hide_code=True)
def _(json, mo, pd, report_path):
    _root = report_path.parent
    _comparison = _root / "wsclean_prior_phase_comparison.png"
    _image = _root / "wsclean_joint_allsky_best_tested_sources.png"
    _metrics_path = _root / "wsclean_prior_phase_comparison.png.json"
    mo.stop(not all(_p.is_file() for _p in (_comparison, _image, _metrics_path)))
    _metrics = json.loads(_metrics_path.read_text())
    _rows = [
        {"Hand": _pol,
         "Before phase RMS (M)": round(_metrics[_pol]["old_residual"]["off_sun_rms"] / 1e6, 3),
         "Latest RMS (M)": round(_metrics[_pol]["new_residual"]["off_sun_rms"] / 1e6, 3),
         "Improvement": f"{100*_metrics[_pol]['residual_rms_reduction_fraction']:.1f}%"}
        for _pol in ("XX", "YY")
    ]
    mo.vstack([
        mo.md("## Latest Sun CLEAN image\n"
              "The phase before/after comparison uses full-band XX "
              "(312–343 MHz) and YY (312–325 MHz). XX uses a 3 m baseline "
              "cut; YY uses 7 m. Both use Briggs 0 and joint multiscale CLEAN. "
              "The comparison has identical image settings and beam on either "
              "side; only the previous-bin antenna phase term changes."),
        mo.ui.table(pd.DataFrame(_rows), pagination=False),
        mo.image(_comparison.read_bytes(), width="100%"),
        mo.md("The best tested map additionally removes 318.4–319.7 MHz "
              "from XX, reducing its off-Sun RMS to 0.310 M. YY retains its "
              "full accepted band at 0.360 M. Source markers and the Galactic "
              "plane are overlays; grating structure remains."),
        mo.image(_image.read_bytes(), width="100%"),
    ])
    return


@app.cell(hide_code=True)
def _(json, mo, pd, report_path):
    _root = report_path.parent
    _plot = _root / "sun_latest_frequency_board_diagnostics.png"
    _phase_fit = _root / "sun_phase_selfcal.npz.json"
    mo.stop(not (_plot.is_file() and _phase_fit.is_file()))
    _fit = json.loads(_phase_fit.read_text())["held_baseline_median"]
    _rows = [
        {"Hand": _pol,
         "Held-out next-bin model error":
             f"{100*_fit[_pol]['odd_bins_before']:.2f}% → "
             f"{100*_fit[_pol]['odd_bins_previous_bin']:.2f}%"}
        for _pol in ("XX", "YY")
    ]
    mo.vstack([
        mo.md("## Frequency and FPGA phase diagnostics\n"
              "Top: error against a moving Sun model on baselines excluded "
              "from the Sun-flux fit. The strongest narrow problem is near "
              "318.5–319.5 MHz. YY also degrades near 324 MHz, and XX degrades "
              "broadly above 334 MHz. Bottom: residual broadband phase shared "
              "by each FPGA relative to FPGA 0. XX includes steps as well as "
              "slow changes; these tracks are not yet evidence of a single "
              "linear clock-rate drift. The phase gains below were fitted "
              "per 10-second bin and tested on the next, unseen bin."),
        mo.ui.table(pd.DataFrame(_rows), pagination=False),
        mo.image(_plot.read_bytes(), width="100%"),
    ])
    return


@app.cell(hide_code=True)
def _(json, mo, pd, report_path):
    _root = report_path.parent
    _trial = _root / "sun_frequency_band_trials.json"
    mo.stop(not _trial.is_file())
    _data = json.loads(_trial.read_text())
    mo.vstack([
        mo.md("## Frequency-cut image trials\n"
              "The narrow lower band has cleaner Sun-model phase, but less "
              "frequency diversity to average away the array sidelobes. "
              "Compare RMS both in image units and relative to the Sun peak. "
              "The lowest-RMS trial is the imaging default; the clean lower "
              "band remains useful for calibration checks."),
        mo.ui.table(pd.DataFrame(_data["rows"]), pagination=False),
    ])
    return


@app.cell(hide_code=True)
def _(mo, report_path):
    _image = report_path.parent / "sun_clock_delay_drift_check.png"
    mo.stop(not _image.is_file())
    mo.vstack([
        mo.md("### Is there an inter-FPGA clock-delay drift?\n"
              "This independent early/middle/late check fits a frequency "
              "slope within each coarse channel and lets each coarse channel "
              "have its own phase offset. XX early-to-late delay changes are "
              "only about −0.10 to +0.18 ns by FPGA pair and close to within "
              "0.03 ns around board triangles. YY pair slopes disagree by up "
              "to 1.75 ns around triangles. A clock-rate correction has not "
              "been validated or applied; the empirical previous-bin "
              "per-antenna phase remains the useful time-varying term."),
        mo.image(_image.read_bytes(), width="100%"),
    ])
    return


@app.cell(hide_code=True)
def _(json, mo, pd, report_path):
    _root = report_path.parent
    _receptors = _root / "receptor_swap_evidence.json"
    _ids = _root / "antenna_id_swap_search.json"
    mo.stop(not (_receptors.is_file() and _ids.is_file()))
    _data = json.loads(_receptors.read_text())["windows"]
    _rows = []
    for _window, _measurements in _data.items():
        for _item in _measurements:
            if _item["antenna_id"] in (8, 11, 12, 17):
                _rows.append({"Window": _window,
                              "Antenna": _item["antenna_id"],
                              "XX crossed/nominal": round(_item["XX"]["swapped_over_nominal"], 2),
                              "YY crossed/nominal": round(_item["YY"]["swapped_over_nominal"], 2)})
    _swaps = json.loads(_ids.read_text())["candidates"]
    _pair = next(_x for _x in _swaps if _x["pair"] == [11, 17])
    mo.vstack([
        mo.md("## Antenna identity check\n"
              "A true hand reversal should make crossed-hand coherence exceed "
              "nominal coherence in both time windows. Antennas 8 and 12 "
              "show that pattern; 11 and 17 do not. An antenna-ID exchange "
              "11↔17 also sharply worsens held-out Sun tracking "
              f"(mean cosine change {_pair['mean_withheld_improvement']:+.3f}). "
              "The weaker 11↔13 ID candidate still needs an independent "
              "direction or hardware trace. No live mapping was changed."),
        mo.ui.table(pd.DataFrame(_rows), pagination=False),
    ])
    return


@app.cell(hide_code=True)
def _(mo, report_path):
    _root = report_path.parent
    _image = _root / "wsclean_joint_allsky_amp_phase_sunpeeled_dirty_sources.png"
    mo.stop(not _image.is_file())
    mo.vstack([
        mo.md("## Moving-Sun-subtracted residual\n"
              "A geometric Sun disk with a fitted real apparent flux per time "
              "and fine channel was removed. Off-Sun dirty-image RMS is "
              "0.313 M XX / 0.295 M YY. This residual has not been CLEANed; "
              "its source markers are positions to test, not detections."),
        mo.image(_image.read_bytes(), width="100%"),
    ])
    return


@app.cell(hide_code=True)
def _(json, mo, pd, report_path):
    _path = report_path.parent / "sun_offsun_source_checks.json"
    _image = report_path.parent / "sun_offsun_source_checks.png"
    mo.stop(not (_path.is_file() and _image.is_file()))
    _results = json.loads(_path.read_text())
    _rows = [{_key: _row[_key] for _key in
              ("Source", "Median split contrast", "Positive splits",
               "Sky-pixel percentile")}
             for _row in _results["rows"]]
    mo.vstack([
        mo.md("## Off-Sun source consistency\n"
              "A source candidate should recur at the same sky position in "
              "independent time and frequency splits, in both hands where "
              "sensitivity permits. The scores below are local image contrasts, "
              "not a formal flux or detection significance. Cen A is positive "
              "in all eight splits, but is extended at this resolution. "
              "No non-Sun point source is yet confirmed."),
        mo.ui.table(pd.DataFrame(_rows), pagination=False),
        mo.image(_image.read_bytes(), width="100%"),
    ])
    return


@app.cell(hide_code=True)
def _(json, mo, pd, report_path):
    _path = report_path.parent / "sun_offsun_fringe_checks.json"
    mo.stop(not _path.is_file())
    _report = json.loads(_path.read_text())
    _rows = [{_key: _value for _key, _value in _row.items()
              if _key in ("Source", "Matched real / Sun", "Matched imaginary / Sun",
                          "Positive splits", "Short-baseline real / Sun",
                          "Long-baseline real / Sun")}
             for _row in _report["rows"] if not _row["control"]]
    _controls = [_row for _row in _report["rows"] if _row["control"]]
    _maximum = max(_row["Matched real / Sun"] for _row in _controls)
    mo.vstack([
        mo.md("### Visibility fringe check\n"
              "Rotate the Sun-subtracted visibilities by each expected "
              "source's geometric fringe and average the real component. "
              "The source-model sign reproduces the independent Sun model "
              f"with coherence {_report['phase_sign_sun_model_coherence']:.6f}. "
              "Cen A has a positive real response in all eight splits; the "
              f"largest of eight nearby control directions reaches {_maximum:.4f} "
              "of the fitted Sun amplitude. Cen A is extended, and this "
              "single capture does not establish a point-source detection."),
        mo.ui.table(pd.DataFrame(_rows), pagination=False),
    ])
    return


if __name__ == "__main__":
    app.run()
