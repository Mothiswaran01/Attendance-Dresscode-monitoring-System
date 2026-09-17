# Troubleshooting

## Camera Issues

| Problem | Likely Cause | Solution |
|---|---|---|
| No frame received | RTSP URL incorrect or camera offline | Verify `CAMERA_SOURCE` env var; test URL with VLC |
| High latency | Buffer overflow or network | Ensure `OPENCV_FFMPEG_CAPTURE_OPTIONS` includes `rtsp_transport;tcp` |
| Connection drops | Unstable network | Add reconnection logic or use a lower frame rate |

## Face Recognition

| Problem | Likely Cause | Solution |
|---|---|---|
| All faces "Unknown" | Embedding database missing or wrong path | Verify `student_embeddings.pkl` exists at `data/` |
| Low similarity scores | Poor quality training images | Use front-facing, well-lit face crops (≥5 per student) |
| Wrong identity matched | Ambiguous embeddings | Retrain with more varied angles of each student |
| No faces detected | Low light or occlusion | Ensure adequate lighting; InsightFace SCRFD works best in good conditions |

## Model Weights

| Problem | Likely Cause | Solution |
|---|---|---|
| YOLO model not loading | Missing `.pt` file | Place weights in `models/dresscode_detection/saved_models/` |
| Pose model missing | `yolov8n-pose.pt` not found | Run `yolo download model=yolov8n-pose.pt` |
| ArcFace ONNX missing | Model not downloaded | InsightFace downloads `buffalo_l` automatically on first `FaceAnalysis()` call |

## Google Sheets

| Problem | Likely Cause | Solution |
|---|---|---|
| Connection failed | Credentials file missing or wrong path | Verify `credentials.json` exists and `GOOGLE_CREDENTIALS_FILE` is correct |
| Permission denied | Sheet not shared with service account | Share the sheet with the `client_email` from your credentials JSON |
| Sheet not found | Wrong sheet name | Verify `GOOGLE_SHEET_NAME` matches the sheet title exactly |
| Slow writes | Network latency | The async queue handles this; check background worker errors in console |

## Performance

| Problem | Likely Cause | Solution |
|---|---|---|
| Low FPS | All models running every frame | Models only run during audit windows (200s bursts) |
| GPU out of memory | Too many models loaded | Reduce model count or use `CPUExecutionProvider` in `attendance_recognition.py` |
| High CPU usage | No GPU available | Install CUDA-enabled PyTorch/ONNX Runtime |

## Common Errors

**`No module named 'insightface'`**
```bash
pip install insightface
```

**`Ultralytics YOLO model not found`**
```bash
pip install ultralytics
```

**`gspread.exceptions.APIError`**
- Check your Google Sheets API quota
- Ensure the service account has Editor access
- Verify the sheet hasn't been deleted or renamed
