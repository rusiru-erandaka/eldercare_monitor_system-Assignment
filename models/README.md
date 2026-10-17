# Local OpenVINO model assets

The model weights in this directory are intentionally **not committed**. The
tracked `models.lock.json` records the exact files used for this project and
their SHA-256 hashes. The pose model came from a local conversion; its original
download URL and redistribution license have not been verified. Do not publish
either model's weights until their licenses have been reviewed.

To populate a fresh checkout from copies you already possess:

```bash
python scripts/fetch_models.py \
  --detector-source /path/to/YOLO11n-fp16-ov-snapshot \
  --pose-source /path/to/yolov8n-pose_openvino_model
```

The script copies only the four listed XML/BIN files. It checks source hashes
before copying and destination hashes afterward, and refuses unrecognized or
corrupted files. No download URL is assumed. The default `eldercare.run`
command verifies these pinned local assets before model compilation. Custom
`--det-model` and `--pose-model` paths remain available and are recorded in the
run manifest, but they are not approved by this pinned manifest.
