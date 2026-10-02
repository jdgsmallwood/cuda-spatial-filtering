#!/usr/bin/env python3
"""Score previously fitted native sky gains on an independent capture."""
import argparse
import json
from pathlib import Path

import numpy as np

from diagnose_sky_calibration import score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fit', type=Path, help='training validation.npz')
    parser.add_argument('transfer', type=Path, help='independent model-only validation.npz')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with np.load(args.fit) as fit, np.load(args.transfer) as transfer:
        frequencies = fit['frequencies_hz']
        order = fit['antenna_order']
        gains = fit['gains']
        good = fit['good']
        joint = fit['joint']
        selected = fit['selected']
        model = transfer['model']
        data = transfer['data']
        transfer_times = transfer['times_jd']
        if not np.array_equal(frequencies, transfer['frequencies_hz']):
            parser.error('training and transfer frequency axes differ')
        if not np.array_equal(order, transfer['antenna_order']):
            parser.error('training and transfer antenna orders differ')
        scores = np.empty((len(frequencies), 2, 3))
        for ci in range(scores.shape[0]):
            for pi in range(2):
                gain = gains[ci, pi]
                for wi in range(3):
                    predicted = model[wi, ci, pi] * gain[:, None] * gain.conj()[None, :]
                    scores[ci, pi, wi] = score(data[wi, ci, pi], predicted, selected)
        report = {'training': str(args.fit), 'transfer': str(args.transfer),
                  'transfer_utc': __import__('astropy.time', fromlist=['Time']).Time(
                      transfer_times, format='jd').isot.tolist(),
                  'scores': {}}
        for pi, pol in enumerate(('XX', 'YY')):
            accepted = good[:, pi]
            report['scores'][pol] = {
                'all_median': np.median(scores[:, pi], axis=0).tolist(),
                'training_accepted_count': int(accepted.sum()),
                'training_accepted_median': np.median(scores[accepted, pi], axis=0).tolist(),
                'training_accepted_fraction_above_0_65':
                    np.mean(scores[accepted, pi] > .65, axis=0).tolist(),
            }
        report['joint_training_accepted_count'] = int(joint.sum())
        report['joint_training_accepted_median'] = {
            pol: np.median(scores[joint, pi], axis=0).tolist()
            for pi, pol in enumerate(('XX', 'YY'))}
    result = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.write_text(result)
        np.savez_compressed(args.output.with_suffix('.npz'), scores=scores,
                            frequencies_hz=frequencies, good=good, joint=joint)
    print(result, end='')


if __name__ == '__main__':
    main()
