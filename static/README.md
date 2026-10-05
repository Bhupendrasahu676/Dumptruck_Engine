# Dump-truck digital twin assets

These files are framework-free browser assets intended for a Streamlit HTML
component or a small `components.html` wrapper:

- `digital_twin.js` exposes `window.DigitalTwinTruck.init` and the alias
  `window.initDumpTruckDigitalTwin`.
- `dashboard.css` contains prefixed (`dt-`) dark/minimal dashboard primitives.

## Basic usage

Load the stylesheet and script in the HTML component, then initialize against a
container element:

```html
<link rel="stylesheet" href="/static/dashboard.css">
<div id="truck-twin" class="dt-container"></div>
<script src="/static/digital_twin.js"></script>
<script>
  const twin = DigitalTwinTruck.init(document.getElementById('truck-twin'), {
    telemetry: {
      payload: 18,          // payload units; maxPayload defaults to 40
      maxPayload: 40,
      temperature: 84,      // °C
      vibration: 2.1,       // normalized values (0..1) or raw / vibrationMax
      hoistAngle: 8,        // degrees; bodyPosition/hoistPosition also work
      speed: 12
    },
    config: {
      showHud: true,
      height: 460,
      // threeUrl can point at an approved/local Three.js asset.
      // useThree: false forces the no-network Canvas 2D renderer.
    }
  });

  // Streamlit can send fresh readings from its rerender/update path.
  twin.update({
    payloadPercent: 72,
    engineTemp: 91,
    vibration: 0.34,
    bodyPosition: 0.45
  });

  // Optional lifecycle helpers:
  // await twin.ready;       // resolves to "three", "fallback", or "disposed"
  // twin.getState();
  // twin.resize();
  // twin.destroy();
</script>
```

`init` also accepts the shorter form `DigitalTwinTruck.init(container,
telemetry, config)`. Telemetry keys are intentionally forgiving: payload/load,
engineTemp/temperature, vibration/vibrationRms, and hoistAngle/bedAngle/body-
or-hoistPosition are recognized.

Three.js is loaded lazily from the configured CDN URL. If the CDN, WebGL, or a
content-security policy blocks it, initialization does not fail: the component
keeps rendering a responsive Canvas 2D visualization with the same payload sag,
temperature color, vibration, and dump-body response. The dashboard HUD can be
disabled with `showHud: false` when a host page supplies its own metric cards.
