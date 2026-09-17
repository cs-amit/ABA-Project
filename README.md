# Smart Sleep

Smart Sleep is a local-first Android wellness prototype for Galaxy Watch4 and later Wear OS powered by Samsung watches. It estimates the likelihood of light sleep within a selected wake window; it does not diagnose sleep disorders or make definitive sleep-stage claims.

## Project modules

- `:app` is the Android phone application and exact-time fallback alarm owner.
- `:wear` is the Wear OS application and Samsung Health Sensor SDK integration point.
- `:core` is the pure Kotlin domain and algorithm module.
- `ml/` contains the Python training and evaluation environment.

## Prerequisites

- JDK 17 and Android SDK platform 36.
- Samsung Galaxy Watch4 or later Samsung-powered Wear OS watch for sensor features.
- Samsung Health Sensor SDK 1.4.1, obtained manually from Samsung, copied locally to `wear/libs/samsung-health-sensor-api.aar`. This binary is intentionally ignored by Git.

Run the Android verification from the repository root:

```powershell
.\gradlew.bat :core:test :app:assembleDebug :wear:assembleDebug
```

For the complete JVM suite, run `.\gradlew.bat test`. The offline ML suite is
`C:\Users\AMIT\anaconda3\envs\fashion-trends\python.exe -m pytest ml/tests -q`.

Current evidence is summarized in [`docs/project-status-2026-09-17.md`](docs/project-status-2026-09-17.md),
the [model card](docs/evaluation/model-card.md), and the [performance-recovery
report](docs/evaluation/bidsleep-performance-recovery-report.md). The Android
app demonstrates capture, transfer, persistence, optional ONNX inference, and
safe fallback alarm behavior. The bundled model remains a classroom research
MVP, not a clinical or deployment-validated model.

The project stores health and session data locally by default. It contains no accounts, cloud sync, analytics, advertising, microphone capture, Samsung Health Data SDK, context logging, or schedule-advisor functionality.

See `docs/development/watch4-setup.md` before using a physical watch.
