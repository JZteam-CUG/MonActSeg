# MonActSeg
## Environment

- Python 3.8+
- PyTorch
- NumPy
- SciPy
- loguru

## Training Data Layout

```
MonActSeg/
  data/
    <dataset>/
      features/
        skeleton/
          <video_id>.npy
        rgb/
          <video_id>.npy
      groundTruth/
        <video_id>.txt
      mapping.txt
      splits/
        train_split_<k>.bundle
        val_split_<k>.bundle
        test_split_<k>.bundle
```

## Training

```bash
python fused_main.py \
  --dataset monkey \
  --split 1
```

## Demo Test

Run the bundled demo with the provided video and precomputed feature files:

```bash
python scripts/infer_from_features.py \
  --video-path demo/demo.mp4 \
  --rgb-feature demo/features/rgb/demo.npy \
  --skeleton-feature demo/features/skeleton/demo.npy
```

Default outputs:

- `outputs/demo_from_features/demo_prediction.txt`
- `outputs/demo_from_features/demo_out.mp4`
