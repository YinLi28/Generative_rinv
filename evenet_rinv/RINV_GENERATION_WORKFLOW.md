# SVJ rinv TruthGeneration Workflow

This workflow keeps runnable files in the main project layout:

- `datasets/svj_rinv_generation.py`: CSV validation and `x`, `conditions`, `x_invisible` construction.
- `scripts/convert_csv_to_rinv_generation_npz.py`: CSV to EveNet NPZ conversion.
- `configs/event_info_rinv_generation.yaml`: EveNet data schema.
- `configs/rinv_generation_EveNet_rinv_test.yaml`: 200-epoch training config for the single-point test.

## Tensor Layout

The visible point cloud follows the EveNet data-preparation doc:

```text
x:      [N, 18, 7]
x_mask: [N, 18]
```

Feature order:

```text
energy, pt, eta, phi, btag_score, is_lepton, charge
```

Slots:

```text
0 = photon
1 = leading jet
2 = subleading jet
3..17 = padding
```

`MET` and `met_phi` are event-level conditions:

```text
conditions: [N, 2] = [MET, met_phi]
```

The target remains a one-token invisible target:

```text
x_invisible:      [N, 1, 1]
x_invisible_mask: [N, 1]
```

`rinv` is scaled with:

```text
rinv_scaled = 2 * (rinv - rinv_min) / (rinv_max - rinv_min) - 1
```

For the current data, use `rinv_min=0` and `rinv_max=1`, so this reduces to:

```text
rinv_scaled = 2 * rinv - 1
```

## Server Commands

Run from:

```bash
cd /home/lyin/evenet_rinv
```

Create output directories:

```bash
mkdir -p /home/lyin/evenet_rinv/experiments/EveNet_rinv_test/{logs,checkpoints,ray_results}
mkdir -p /group-data/svj/EveNet_generative/EveNet_rinv_test/{npz,parquet,parquet_train,parquet_val,parquet_test}
```

Convert CSV to NPZ:

```bash
PYTHONPATH=/home/lyin/evenet_rinv \
python3.12 scripts/convert_csv_to_rinv_generation_npz.py \
  --csv /home/lyin/mMaos_reco/Gamma500_data/csv_files_sig/zmass1500_0p5.csv \
  --out-dir /group-data/svj/EveNet_generative/EveNet_rinv_test/npz \
  --report-path /group-data/svj/EveNet_generative/EveNet_rinv_test/npz/report.json \
  --rinv-min 0 \
  --rinv-max 1
```

Convert NPZ to parquet:

```bash
PYTHONPATH=/home/lyin/evenet_rinv/pythonpath:/home/lyin/evenet_rinv/repos/EveNet-Full \
python3.12 repos/EveNet-Full/preprocessing/preprocess.py \
  --config configs/rinv_generation_EveNet_rinv_test.yaml \
  --train /group-data/svj/EveNet_generative/EveNet_rinv_test/npz/rinv_generation_train.npz \
  --val /group-data/svj/EveNet_generative/EveNet_rinv_test/npz/rinv_generation_val.npz \
  --test /group-data/svj/EveNet_generative/EveNet_rinv_test/npz/rinv_generation_test.npz \
  --store_dir /group-data/svj/EveNet_generative/EveNet_rinv_test/parquet \
  -v
```

Prepare EveNet split directories:

```bash
cp /group-data/svj/EveNet_generative/EveNet_rinv_test/parquet/train.parquet /group-data/svj/EveNet_generative/EveNet_rinv_test/parquet_train/
cp /group-data/svj/EveNet_generative/EveNet_rinv_test/parquet/val.parquet /group-data/svj/EveNet_generative/EveNet_rinv_test/parquet_val/
cp /group-data/svj/EveNet_generative/EveNet_rinv_test/parquet/test.parquet /group-data/svj/EveNet_generative/EveNet_rinv_test/parquet_test/
cp /group-data/svj/EveNet_generative/EveNet_rinv_test/parquet/shape_metadata.json /group-data/svj/EveNet_generative/EveNet_rinv_test/parquet_train/
cp /group-data/svj/EveNet_generative/EveNet_rinv_test/parquet/shape_metadata.json /group-data/svj/EveNet_generative/EveNet_rinv_test/parquet_val/
cp /group-data/svj/EveNet_generative/EveNet_rinv_test/parquet/shape_metadata.json /group-data/svj/EveNet_generative/EveNet_rinv_test/parquet_test/
```

Train:

```bash
nohup bash -lc 'PYTHONPATH=/home/lyin/evenet_rinv/pythonpath:/home/lyin/evenet_rinv/repos/EveNet-Full CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline WANDB_API_KEY=dummy python3.12 repos/EveNet-Full/scripts/train.py configs/rinv_generation_EveNet_rinv_test.yaml --ray_dir /home/lyin/evenet_rinv/experiments/EveNet_rinv_test/ray_results' \
  > /home/lyin/evenet_rinv/experiments/EveNet_rinv_test/logs/train.log 2>&1 &
```
