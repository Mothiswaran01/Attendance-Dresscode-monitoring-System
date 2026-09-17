# System Architecture

## High-Level Flow

```
Camera (RTSP)
    │
    ▼
LatestFrameReader (background thread, always reads latest frame)
    │
    ▼ (every 100ms)
┌────────────────────────────────────────────────────┐
│ Main Processing Loop                                │
│                                                     │
│  1. Face Recognition (InsightFace)                  │
│     ├─ SCRFD detection → ArcFace embedding          │
│     └─ Cosine similarity vs database                 │
│                                                     │
│  2. Temporal Buffer (7 frames, min 4 votes)         │
│                                                     │
│  3. If AUDIT MODE active:                           │
│     ├─ ID Card detection (YOLO)                     │
│     ├─ Shoe detection (YOLO)                        │
│     └─ Tuck-in estimation (pose keypoints)          │
│                                                     │
│  4. If audit window ended:                          │
│     └─ Log results to Google Sheets                 │
└────────────────────────────────────────────────────┘
```

## Audit Mechanism

The system uses periodic **audit bursts** to determine attendance and compliance:

```
[IDLE]────[AUDIT 200s]────[IDLE]────[AUDIT 200s]────[IDLE]──→
           ↑                              ↑
      Startup audit              Session interval change
```

- **Idle mode**: Face recognition runs but no logging occurs
- **Audit mode**: 200-second window where detections are counted
- Students with ≥15 face hits (out of ~300 frames) are approved
- Dress code items need ≥10 hits each

## Session Schedule

Default schedule (7 sessions):

| Session | Start | End |
|---|---|---|
| 1 | 09:20 | 10:10 |
| 2 | 10:10 | 11:00 |
| 3 | 11:15 | 12:05 |
| 4 | 12:05 | 12:50 |
| 5 | 13:45 | 14:40 |
| 6 | 14:40 | 15:35 |
| 7 | 15:35 | 16:30 |

Each session is divided into 3 intervals (~16–18 min each). An audit burst runs at the start of each interval.

## Temporal Buffer

Prevents identity flickering using sliding window majority voting:

```
Frames: ┌───┬───┬───┬───┬───┬───┬───┐
         │ A │ A │ B │ A │ A │ B │ A │
         └───┴───┴───┴───┴───┴───┴───┘
                      Majority: A (5/7)
                      Min votes: 4 → Confirmed: A
```

- Buffer size: 7 frames
- Min votes required: 4
- Ignores "Unknown" votes
- Returns `None` until minimum votes are met

## Threading Model

```
┌────────────────────┐
│ Camera Reader      │  ← Dedicated thread (always reads latest frame)
│ (LatestFrameReader) │
└────────┬───────────┘
         │
┌────────▼───────────┐
│ Main Loop          │  ← Main/Flask thread (processes every 100ms)
│ (face + dresscode) │
└────────┬───────────┘
         │
┌────────▼───────────┐
│ Google Sheets      │  ← Background daemon thread (async queue)
│ Logger Worker      │
└────────────────────┘
```

## Key Directories

| Path | Purpose |
|---|---|
| `data/student_embeddings.pkl` | Face embedding database |
| `data/processed/dresscode_frames/` | Dress code proof frames |
| `models/face_recognition/saved_models/` | ArcFace ONNX model |
| `models/dresscode_detection/saved_models/` | YOLO dress code models |
| `logs/` | Application logs (if configured) |
