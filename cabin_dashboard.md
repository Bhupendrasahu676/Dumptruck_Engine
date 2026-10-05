# Cabin dashboard

Run the responsive Android/tablet-style cabin console from the `Engine`
directory:

```bash
streamlit run cabin_dashboard.py --server.port 8502
```

Open `http://localhost:8502` on the host or on a tablet/Android device that can
reach the host. Use the sidebar to select the scenario, fixed payload class,
start condition, and five fault toggles. The dashboard loads one current
telemetry frame, then replays or steps through the deterministic journey in the
browser. The 3D twin uses `static/digital_twin.js` with its built-in Canvas
fallback when WebGL/CDN access is unavailable.
