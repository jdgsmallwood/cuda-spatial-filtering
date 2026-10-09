#!/usr/bin/env python3
"""Assemble the 2026-10-04 stationary-interference and calibration evidence."""
import base64
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "results/night_delay_audit_20261004"


def main():
    def read(path):
        return json.loads((ROOT / path).read_text())
    projection = read("post_reboot_stationary_projection/report.json")
    pn = read("pn_fits/post_reboot_full40_slopes/report.json")
    transfer = read("post_reboot_pn_sky_transfer/report.json")
    joint = read("post_reboot_joint_sky_stationary/report.json")
    inputs = read("polarization_input_audit/report.json")
    live = read("polarization_input_audit/live_map_verification.json")
    source = read("source_expectations.json")
    control = read("post_reboot_antenna_switch/antenna_switch_control.json")
    sections = [
        dict(title="One stationary eigenmode", file="post_reboot_stationary_projection/stationary_projection.png",
            explanation="Left: cross-correlation RMS on the later half of the observation. Green subtracts a single covariance mode learned from early data, with one real amplitude per later integration. Orange projects out a voltage mode and also changes the receiver-noise covariance. Right: temporal predictions fitted only to early data. These are visibility diagnostics, not image RMS or source detections. The lower band is a deliberate solar-signal loss control."),
        dict(title="Raw packets: antennas on/off", file="post_reboot_antenna_switch/antenna_switch_control.png",
            explanation="Packet scales are included. Turning the antennas off reduces upper-band YY coherence by about 30 times. The coherent signal exists before GPU processing and enters upstream of the antenna selection switch. This does not separate external RFI from pickup in the antenna/front-end path."),
        dict(title="Fresh PN delay measurements", file="pn_fits/post_reboot_full40_slopes/pn_fullband_slopes.png",
            explanation="The top panel tests a fitted phase slope on coarse channels excluded from fitting. The bottom compares slopes between coarse centres and within a coarse channel. Two PN toggles used the same verified FPGA boot and 400–439 tuning. Their difference is an effective response; the PN comb does not uniquely identify a physical pipeline delay or remove delay aliases by itself."),
        dict(title="PN correction followed by a sky residual", file="post_reboot_pn_sky_transfer/pn_sky_transfer.png",
            explanation="Fit the first four coarse channels in the early half; predict other frequencies in the later half. PN reduces the median remaining fitted delay from 335 to 18 ns and improves transfer, especially across boards. The reversed-PN control also fits fairly well when a large residual delay is allowed. This comparison therefore does not independently establish the PN sign or unique physical delay."),
        dict(title="Sun + catalogue + stationary component", file="post_reboot_joint_sky_stationary/joint_sky_stationary.png",
            explanation="Stable antenna gains and an additive per-baseline constant were fitted only to early data. Later data decide whether the model predicts anything. YY has a useful moving solar component; adding the point-source catalogue gives little further improvement. This is not a Fornax A detection. XX remains poorly described by the current moving-sky model."),
    ]
    report = dict(schema="starspin_stationary_followup_v1", date="2026-10-04",
        title="Stationary interference, input mapping and realistic sky targets",
        evidence_root=str(ROOT), sections=sections,
        capture=projection["capture"], projection=projection["medians"],
        raw_control=control["summary"], noise_calibration=pn,
        packet_scale_check=read("post_reboot_raw_scale_check.json"),
        sky_transfer=transfer["scores"], joint_sky=joint["summary"],
        input_mapping=inputs, live_verification=live, source_expectations=source,
        array=dict(antennas=36, longest_baseline_m=34.240067421078486,
            shortest_baseline_m=1.7433510834023078,
            lambda_over_longest_baseline_at_330MHz_deg=1.5201791941777418,
            approximate_largest_scale_at_330MHz_with_5m_cut_deg=6.246124572091995),
        changes=["Removed the pre-existing 8/12 hand reversals from the live input map; raw even/odd now map to canonical 0/1 for all connected antennas.",
            "Added captured-map provenance to native sky-calibration reports to distinguish input reversals from additional offline reversals.",
            "Added offline mode-removal, joint sky/stationary and fresh-PN transfer diagnostics. No interference projection or new gain table was installed in the live pipeline."],
        tests=["Rebuilt GPU complex CPU-reference and individual-polarization-swap tests: 2 passed.",
            "Synthetic noise preservation, injected sky response, held-time isolation and gain recovery: 3 tests passed."],
        current_hardware="Four boards, 400–439, gain setting 20, antennas on, PN off. Delay closure rechecked before and after the PN controls; maximum pair discrepancy one sample. No capture left running.",
        expectations=["Fornax A detection is realistic: the local catalogue predicts about 379 Jy at 330 MHz before beam attenuation; detailed radio-lobe structure is below this array's approximate 1.5-degree resolution.",
            "During the saved low-Sun capture the current beam model predicts roughly 131–164 Jy apparent YY flux for Fornax A; this depends on the unverified beam model.",
            "A recognisable Galactic plane is a reasonable separate goal with short baselines and beam modelling. The 5 m cut removes sensitivity to broad structure (rough scale 6 degrees at 330 MHz); cross-correlations cannot measure the all-sky monopole.",
            "No calibrated SEFD or independently validated absolute image noise floor is available here, so no absolute detection significance is claimed.",
            "A dark observation near Fornax A transit provides a much simpler source-detection test than peeling a bright Sun. Diffuse Galactic imaging requires its own short-baseline validation."],
        beam_check="The saved observation had the Sun only 16–19 degrees high, nearly due east. At 330 MHz the current element-beam model predicts XX power response 0.0028–0.0081 versus YY 0.154–0.230: the Sun should be about 29–56 times weaker in XX. A poor XX Sun fit at this elevation alone does not establish a failed hand. The beam prediction still needs observational validation.",
        hardware_status_note="FPGA2 status retained an ADC16–19 group overflow/underflow latch (link bit 2), present both before and after these controls. This latch does not date an event or establish an ongoing error. Fresh packet scales were all 1 on all four boards, with about 95% nonzero IQ components; no recurrence of the 64/128 scale anomaly was found in these sampled controls.",
        references=[dict(title="Fornax A low-frequency flux measurements (McKinley et al.)", url="https://arxiv.org/abs/1411.1487"),
            dict(title="NRAO: interferometric resolution and missing short spacings", url="https://almascience.nrao.edu/about-alma/alma-basics")])
    path = ROOT / "stationary_followup_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    body = ["<!doctype html><meta charset='utf-8'><title>Starspin calibration follow-up</title>",
        "<style>body{font:17px system-ui;max-width:1250px;margin:35px auto;padding:0 20px;color:#17212b;line-height:1.5}img{max-width:100%}h2{margin-top:2em}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px;text-align:left}.note{background:#eef4fa;padding:18px;border-radius:8px}</style>",
        "<h1>Stationary interference and calibration</h1><p>4 October 2026</p>",
        "<div class='note'><b>Assessment:</b> A bright Fornax A detection is a realistic target. A recognisable Galactic plane needs short baselines and a more complete beam/sky model. Neither has yet been demonstrated by these tests.</div>",
        "<h2>What changed</h2><ul>" + "".join("<li>" + html.escape(t) + "</li>" for t in report["changes"]) + "</ul>",
        "<p>The 60-second live map check processed 61 integrations, with " +
        f"{live['missing_percent']:.3f}% missing samples overall (0.5% aggregate gate); the worst integration was {live['max_missing_percent']:.2f}%. " +
        "That smoke test verified the mapping; the science diagnostics use the earlier 15-minute capture with 0.01085% overall missing samples and select only zero-loss integrations.</p>",
        "<h2>Realistic targets</h2><ul>" + "".join("<li>" + html.escape(t) + "</li>" for t in report["expectations"]) + "</ul>"]
    body.append("<h2>Why XX is weak in this capture</h2><p>" + html.escape(report["beam_check"]) + "</p>")
    for section in sections:
        encoded = base64.b64encode((ROOT / section["file"]).read_bytes()).decode()
        body.extend(["<h2>" + html.escape(section["title"]) + "</h2>",
            "<p>" + html.escape(section["explanation"]) + "</p>",
            "<img src='data:image/png;base64," + encoded + "'>"])
    body.extend(["<h2>Verification and final state</h2><ul>" + "".join("<li>" + html.escape(t) + "</li>" for t in report["tests"]) + "</ul>",
        "<p>" + html.escape(report["current_hardware"]) + "</p>",
        "<p>" + html.escape(report["hardware_status_note"]) + "</p>",
        "<p>References: " + ", ".join("<a href='" + t["url"] + "'>" + html.escape(t["title"]) + "</a>" for t in report["references"]) + "</p>"])
    (ROOT / "stationary_followup_report.html").write_text("\n".join(body))
    print(path)


if __name__ == "__main__":
    main()
