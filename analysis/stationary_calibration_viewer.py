"""Latest stationary-interference and calibration audit, with a report picker."""
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


@app.cell
def _(Path, mo):
    _root = Path("/data/BLACKMESA_1/sma075/cuda-spatial-filtering/analysis/results")
    report_picker = mo.ui.file_browser(initial_path=_root, filetypes=[".json"],
        multiple=False, restrict_navigation=False, label="Choose a stationary follow-up report")
    report_path_text = mo.ui.text(value=str(_root / "night_delay_audit_20261004/stationary_followup_report.json"),
        full_width=True, label="Or paste the report path")
    mo.vstack([report_picker, report_path_text])
    return report_path_text, report_picker


@app.cell
def _(Path, json, mo, report_path_text, report_picker):
    _picked = report_picker.path(index=0) if report_picker.value else report_path_text.value
    _path = Path(_picked).expanduser()
    mo.stop(not _path.is_file(), mo.callout("Choose an existing report.", kind="warn"))
    audit_report = json.loads(_path.read_text())
    mo.stop(audit_report.get("schema") != "starspin_stationary_followup_v1",
            mo.callout("Choose stationary_followup_report.json.", kind="warn"))
    evidence_root = _path.parent
    return audit_report, evidence_root


@app.cell(hide_code=True)
def _(audit_report, mo):
    mo.vstack([mo.md("# Stationary interference and calibration\n\n" +
        "**Fornax A is a realistic detection target.** A recognisable Galactic plane "
        "requires the short baselines and a better beam model. These tests have "
        "not yet established either detection."),
        mo.md("## What changed\n\n" + "\n".join("- " + _t for _t in audit_report["changes"]))])
    return


@app.cell(hide_code=True)
def _(audit_report, mo, pd):
    _rows = []
    for _pol, _regions in audit_report["projection"].items():
        for _region, _result in _regions.items():
            _rows.append({"Hand": _pol, "Region": _region,
                "Sampled fine bins": _result["channels"],
                "Mode subtraction RMS / original": round(_result["component_subtraction_cross_rms_ratio"], 3),
                "Spatial projection RMS / original": round(_result["cross_rms_ratio"], 3)})
    mo.vstack([mo.md("## Held-data results\n\nThese are visibility RMS ratios, not image noise. "
        "The direction of the removed mode comes only from the first time half; "
        "covariance subtraction fits one real amplitude per later integration. "
        "Lower-band solar suppression is a failure control."),
        mo.ui.table(pd.DataFrame(_rows), pagination=False)])
    return


@app.cell(hide_code=True)
def _(audit_report, evidence_root, mo):
    _sections = []
    for _section in audit_report["sections"]:
        _image = evidence_root / _section["file"]
        _sections.append(mo.vstack([mo.md("## " + _section["title"] + "\n\n" + _section["explanation"]),
            mo.image(_image.read_bytes(), width="100%") if _image.is_file()
            else mo.callout("Missing plot: " + str(_image), kind="warn")]))
    mo.vstack(_sections)
    return


@app.cell(hide_code=True)
def _(audit_report, mo):
    _live = audit_report["live_verification"]
    mo.vstack([mo.md("## What we should be able to see\n\n" +
        "\n".join("- " + _t for _t in audit_report["expectations"])),
        mo.md("## Why XX is weak in this capture\n\n" + audit_report["beam_check"]),
        mo.md("## Live verification\n\n" +
            f"60 s; {_live['integrations']} integrations; **{_live['missing_percent']:.3f}% missing** "
            f"overall, worst integration {_live['max_missing_percent']:.2f}%. "
            "The aggregate gate was 0.5%. The offline science tests used only "
            "zero-loss integrations from the earlier 15-minute capture.\n\n" +
            audit_report["current_hardware"] + "\n\n" + audit_report["hardware_status_note"]),
        mo.accordion({"Full numeric audit": mo.json(audit_report)})])
    return


if __name__ == "__main__":
    app.run()
