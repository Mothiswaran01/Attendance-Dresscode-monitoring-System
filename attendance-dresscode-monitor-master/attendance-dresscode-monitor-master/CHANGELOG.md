# Changelog

## [1.0.0] - 2026-07-05

### Added
- Initial release
- Face recognition using InsightFace (SCRFD + ArcFace)
- Temporal buffer with majority voting for stable identity
- YOLO-based dress code detection (ID card, shoes, tuck-in)
- Pose-based tuck-in estimation using keypoint analysis
- Audit burst mechanism (200s windows per session interval)
- Google Sheets integration with async logging worker
- Desktop GUI mode (`main.py`) with OpenCV display
- Flask streaming server mode (`main_stream.py`) with MJPEG feed
- Session schedule configuration (7 periods)
- Face embedding database generation pipeline
- Video-to-frames face extraction utility
