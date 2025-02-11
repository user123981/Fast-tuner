from pathlib import Path
import json

import numpy as np
from skimage import io
from skimage import transform


ids_fns = 'path/to/json'
ids_fns = Path(ids_fns)
base_path = ids_fns.parent

target_path = Path('./_data/SLO-FAF/train/')

with open(ids_fns, 'r') as f:
    ids_fns = json.load(f)

for p, samples in ids_fns.items():
    print(p, len(samples))
    for s in samples:
        file_set_id = s['FileSetId']
        print(file_set_id)
        faf_fn = base_path / file_set_id / 'FAF_registration' / f'faf.{file_set_id}.png'
        slo_fn = base_path / file_set_id / f'slo.{file_set_id}.png'
        assert faf_fn.exists()
        assert slo_fn.exists()
        faf = io.imread(faf_fn)
        slo = io.imread(slo_fn)
        resize_params = dict(
            output_shape=(768, 768),
            preserve_range=True,
            anti_aliasing=True,
            order=3,
        )
        faf = transform.resize(faf, **resize_params)  # type: ignore
        slo = transform.resize(slo, **resize_params)  # type: ignore
        faf = faf.astype(np.uint8)
        slo = slo.astype(np.uint8)
        target_faf_fn = target_path / 'faf' / 'all' / f'{file_set_id}.png'
        target_slo_fn = target_path / 'slo' / 'all' / f'{file_set_id}.png'
        assert not target_faf_fn.exists()
        assert not target_slo_fn.exists()
        io.imsave(target_faf_fn, faf)
        io.imsave(target_slo_fn, slo)

