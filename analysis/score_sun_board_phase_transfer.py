#!/usr/bin/env python3
"""Compare previous-bin FPGA-common and antenna phase corrections."""

import argparse
import json
from pathlib import Path

from casacore.tables import table
import numpy as np

from fit_sun_amplitude_selfcal import score


def project_to_boards(phase, supported, board_of):
    result = np.zeros_like(phase)
    for board in range(4):
        ids = np.flatnonzero(board_of == board)
        good = ids[supported[ids]]
        if len(good):
            mean = np.angle(np.mean(np.exp(1j * phase[good])))
            result[ids] = mean
    return result


def main(source: Path, model_path: Path, fit_path: Path,
         config_path: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    with np.load(fit_path) as fit:
        phases = fit["phase"]
        supported = fit["supported"]
    config = json.loads(config_path.read_text())
    board_dict = {int(ant): int(board)
                  for board, ants in config["antenna_positions"].items()
                  for ant in ants}
    board_of = np.array([board_dict[ant] for ant in range(1, 37)])
    results = {pol: {kind: [] for kind in
                     ("all_before", "all_board", "all_antenna",
                      "cross_before", "cross_board", "cross_antenna",
                      "same_before", "same_board", "same_antenna")}
               for pol in ("XX", "YY")}
    with table(str(source), ack=False) as data, table(str(model_path), ack=False) as model:
        a = data.getcol("ANTENNA1", 0, 630)
        b = data.getcol("ANTENNA2", 0, 630)
        length = np.linalg.norm(data.getcol("UVW", 0, 630), axis=1)
        for pi, pol in enumerate(("XX", "YY")):
            cut = 3 if pol == "XX" else 7
            long = np.flatnonzero(length > cut)
            train = long[np.arange(len(long)) % 5 != 0]
            held = long[np.arange(len(long)) % 5 == 0]
            for ti in range(1, len(phases), 2):
                vis = data.getcol("DATA", ti * 630, 630)[:, :, pi:pi+1]
                pred = model.getcol("MODEL_DATA", ti * 630, 630)[:, :, pi:pi+1]
                flag = data.getcol("FLAG", ti * 630, 630)[:, :, pi:pi+1]
                board_phase = project_to_boards(phases[ti-1, pi],
                                                supported[ti-1, pi], board_of)
                corrections = {"before": np.ones(630),
                               "board": np.exp(1j * (board_phase[a] - board_phase[b])),
                               "antenna": np.exp(1j * (phases[ti-1, pi, a] -
                                                       phases[ti-1, pi, b]))}
                for group, group_held in (
                        ("all", held),
                        ("cross", held[board_of[a[held]] != board_of[b[held]]]),
                        ("same", held[board_of[a[held]] == board_of[b[held]]])):
                    if not len(group_held):
                        continue
                    for name, correction in corrections.items():
                        value = score(vis / correction[:, None, None], pred,
                                      flag, train, group_held)[0]
                        results[pol][f"{group}_{name}"].append(value)
    summary = {pol: {name: float(np.nanmedian(values))
                     for name, values in groups.items()}
               for pol, groups in results.items()}
    output.write_text(json.dumps({"method": "fit even bins, predict next odd bins; "
                                  "one fifth of long baselines withheld; "
                                  "board phase is circular mean of antenna fits",
                                  "source": str(source), "summary": summary},
                                 indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("fit", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.source, args.model, args.fit, args.config, args.output)
