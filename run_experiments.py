"""Run ten independent training processes concurrently (exclude BPR + real)."""
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    output = ROOT / 'runs' / datetime.now().strftime('parallel-10-%Y%m%d-%H%M%S')
    output.mkdir(parents=True)
    (output / 'train_snapshot.py').write_bytes((ROOT / 'train.py').read_bytes())
    manifest = {'python': sys.executable, 'train_sha256': hashlib.sha256((ROOT / 'train.py').read_bytes()).hexdigest(),
                'data_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (ROOT / 'data/processed').glob('*.csv')}, 'jobs': []}
    processes = []
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    for model in ('mf', 'ncf'):
        for loss in ('bce', 'mse', 'bpr'):
            for mode in ('real', 'random'):
                if loss == 'bpr' and mode == 'real':
                    continue
                name = f'{model}-{loss}-' + ('sampling' if mode == 'random' else 'real')
                folder = output / name
                folder.mkdir()
                command = [sys.executable, str(output / 'train_snapshot.py'), '--data-dir', str(ROOT / 'data/processed'),
                           '--model', model, '--loss', loss, '--neg-mode', mode,
                           '--threads', '1', '--output-dir', str(folder)]
                log = (folder / 'train.log').open('w')
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                job = dict(name=name, command=command, pid=process.pid)
                manifest['jobs'].append(job)
                processes.append((process, log, job, folder))
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Started {len(processes)} parallel jobs: {output}', flush=True)
    results = []
    for process, log, job, folder in processes:
        job['exit_code'] = process.wait()
        log.close()
        if job['exit_code'] == 0:
            metrics = json.loads((folder / 'metrics.json').read_text())
            results.append(dict(experiment=job['name'], best_epoch=metrics['best_epoch'],
                                epochs_run=metrics['epochs_run'], validation_loss=metrics['best_validation_loss'],
                                test_loss=metrics['test_loss'], stopped_early=metrics['stopped_early']))
        print(f"{job['name']}: exit {job['exit_code']}", flush=True)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (output / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
    if results:
        with (output / 'summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
    if len(results) != 10:
        raise RuntimeError(f'Only {len(results)}/10 experiments succeeded; inspect train.log')
    print(f'PASS: 10/10 completed; summary: {output / "summary.csv"}', flush=True)


if __name__ == '__main__':
    main()
