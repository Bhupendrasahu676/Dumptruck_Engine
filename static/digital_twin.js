/*
 * Dump-truck digital twin
 *
 * A dependency-free browser asset.  Three.js is loaded lazily from a CDN when
 * it is available; a small Canvas 2D renderer is kept as a no-network
 * fallback, which is useful in Streamlit deployments with restrictive CSPs.
 *
 * Public API:
 *   const twin = DigitalTwinTruck.init(container, {
 *     telemetry: { payload: 18, maxPayload: 40, temperature: 84,
 *                  vibration: 0.2, hoistAngle: 8 },
 *     config: { showHud: true }
 *   });
 *   twin.update(nextTelemetry);
 *   twin.destroy();
 *
 * The same initializer is also exposed as window.initDumpTruckDigitalTwin.
 */
(function exposeDigitalTwin(global) {
  'use strict';

  var DEFAULT_CONFIG = {
    threeUrl: 'https://cdn.jsdelivr.net/npm/three@0.161.0/build/three.min.js',
    threeLoadTimeout: 7000,
    useThree: true,
    clearContainer: false,
    showHud: true,
    height: 460,
    background: '#071017',
    bodyColor: '#d78b2d',
    bodyDarkColor: '#8a531e',
    accentColor: '#f3b548',
    maxPayload: 40,
    maxSuspensionSag: 0.30,
    maxBedAngle: 34,
    vibrationMax: 10,
    vibrationAmplitude: 0.040,
    vibrationFrequency: 18,
    temperatureRange: [40, 120],
    cameraFov: 42,
    cameraDistance: 9.2,
    cameraHeight: 5.4,
    cameraTargetY: 1.0,
    pixelRatioMax: 2,
    idleTemperature: 72
  };

  var pendingThreeLoad = null;

  function isObject(value) {
    return value !== null && typeof value === 'object';
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function numberOr(value, fallback) {
    if (typeof value === 'number' && isFinite(value)) {
      return value;
    }
    if (typeof value === 'string' && value.trim() !== '') {
      var parsed = Number(value.replace(/[^0-9+\-.eE]/g, ''));
      if (isFinite(parsed)) {
        return parsed;
      }
    }
    return fallback;
  }

  function unwrap(value) {
    if (isObject(value)) {
      if (value.value !== undefined) return value.value;
      if (value.reading !== undefined) return value.reading;
      if (value.current !== undefined) return value.current;
    }
    return value;
  }

  function readPath(source, path) {
    if (!isObject(source)) return undefined;
    var parts = path.split('.');
    var value = source;
    for (var i = 0; i < parts.length; i += 1) {
      if (value === null || value === undefined) return undefined;
      value = value[parts[i]];
    }
    return unwrap(value);
  }

  function sourcesFor(raw) {
    var source = isObject(raw) ? raw : {};
    var list = [source, source.telemetry, source.sensors, source.data];
    return list.filter(function (entry, index) {
      return isObject(entry) && list.indexOf(entry) === index;
    });
  }

  function findReading(raw, keys) {
    var sources = sourcesFor(raw);
    for (var i = 0; i < sources.length; i += 1) {
      for (var j = 0; j < keys.length; j += 1) {
        var value = readPath(sources[i], keys[j]);
        if (value !== undefined && value !== null && value !== '') {
          return { value: value, key: keys[j] };
        }
      }
    }
    return null;
  }

  function fraction(value) {
    var parsed = numberOr(value, 0);
    return clamp(Math.abs(parsed) > 1 ? parsed / 100 : parsed, 0, 1);
  }

  function temperatureColorCss(temperature, range) {
    var low = range[0];
    var high = range[1];
    var normalized = clamp((temperature - low) / (high - low || 1), 0, 1);
    /* HSL travels blue -> cyan -> green -> yellow -> red. */
    var hue = (1 - normalized) * 0.66;
    return 'hsl(' + Math.round(hue * 360) + ', 86%, 55%)';
  }

  function temperatureColorThree(THREE, temperature, range) {
    var low = range[0];
    var high = range[1];
    var normalized = clamp((temperature - low) / (high - low || 1), 0, 1);
    var color = new THREE.Color();
    color.setHSL((1 - normalized) * 0.66, 0.86, 0.55);
    return color;
  }

  function mergeConfig(config) {
    var merged = {};
    Object.keys(DEFAULT_CONFIG).forEach(function (key) {
      merged[key] = DEFAULT_CONFIG[key];
    });
    if (isObject(config)) {
      Object.keys(config).forEach(function (key) {
        if (config[key] !== undefined) merged[key] = config[key];
      });
    }
    if (!Array.isArray(merged.temperatureRange) || merged.temperatureRange.length < 2) {
      merged.temperatureRange = DEFAULT_CONFIG.temperatureRange.slice();
    }
    merged.temperatureRange = [
      numberOr(merged.temperatureRange[0], DEFAULT_CONFIG.temperatureRange[0]),
      numberOr(merged.temperatureRange[1], DEFAULT_CONFIG.temperatureRange[1])
    ];
    if (merged.temperatureRange[1] <= merged.temperatureRange[0]) {
      merged.temperatureRange[1] = merged.temperatureRange[0] + 1;
    }
    return merged;
  }

  function deriveTelemetry(raw, config) {
    var payloadPercent = findReading(raw, [
      'payloadPercent', 'payload_pct', 'loadPercent', 'load_pct',
      'payloadRatio', 'loadRatio', 'payloadFraction', 'loadFraction'
    ]);
    var payloadReading = findReading(raw, [
      'payload', 'payloadTons', 'payload_tons', 'load', 'loadTons', 'load_tons',
      'mass', 'weight'
    ]);
    var maxPayloadReading = findReading(raw, [
      'maxPayload', 'max_payload', 'payloadCapacity', 'capacity', 'ratedPayload'
    ]);

    var payload = payloadReading ? numberOr(payloadReading.value, 0) : 0;
    var maxPayload = maxPayloadReading
      ? numberOr(maxPayloadReading.value, config.maxPayload)
      : numberOr(config.maxPayload, 40);
    var loadRatio;
    if (payloadPercent) {
      loadRatio = fraction(payloadPercent.value);
    } else if (payloadReading && Math.abs(payload) <= 1) {
      loadRatio = fraction(payload);
    } else {
      loadRatio = maxPayload > 0 ? clamp(payload / maxPayload, 0, 1) : 0;
    }

    var temperatureReading = findReading(raw, [
      'temperature', 'engineTemperature', 'engineTemp', 'coolantTemperature',
      'coolantTemp', 'hydraulicTemperature', 'hydraulicTemp', 'temp'
    ]);
    var temperature = temperatureReading
      ? numberOr(temperatureReading.value, config.idleTemperature)
      : numberOr(config.idleTemperature, 72);

    var vibrationPercent = findReading(raw, [
      'vibrationPercent', 'vibration_pct', 'shakePercent', 'vibrationRatio'
    ]);
    var vibrationReading = findReading(raw, [
      'vibration', 'vibrationRms', 'vibration_rms', 'vibrationLevel',
      'shake', 'rmsVibration'
    ]);
    var vibration = vibrationPercent
      ? fraction(vibrationPercent.value)
      : (vibrationReading
        ? clamp(Math.abs(numberOr(vibrationReading.value, 0)) /
          (Math.abs(numberOr(vibrationReading.value, 0)) <= 1
            ? 1
            : numberOr(config.vibrationMax, 10)), 0, 1)
        : 0);

    var angleReading = findReading(raw, [
      'hoistAngle', 'hoist_angle', 'bedAngle', 'bed_angle', 'bodyAngle', 'dumpAngle'
    ]);
    var positionReading = findReading(raw, [
      'hoistPosition', 'hoist_position', 'bodyPosition', 'body_position',
      'bedPosition', 'bed_position', 'hoistPercent', 'bodyPercent'
    ]);
    var maxBedAngle = numberOr(config.maxBedAngle, 34);
    var hoistAngle = 0;
    if (angleReading) {
      hoistAngle = clamp(Math.abs(numberOr(angleReading.value, 0)), 0, maxBedAngle);
    } else if (positionReading) {
      hoistAngle = fraction(positionReading.value) * maxBedAngle;
    }

    var speedReading = findReading(raw, ['speed', 'speedKph', 'speed_kph', 'vehicleSpeed']);
    var rpmReading = findReading(raw, ['rpm', 'engineRpm', 'engine_rpm']);
    var timestampReading = findReading(raw, ['timestamp', 'time', 'updatedAt']);

    return {
      load: loadRatio,
      payload: payload,
      maxPayload: maxPayload,
      temperature: temperature,
      temperatureNorm: clamp(
        (temperature - config.temperatureRange[0]) /
          (config.temperatureRange[1] - config.temperatureRange[0] || 1),
        0,
        1
      ),
      vibration: vibration,
      hoistAngle: hoistAngle,
      hoistNorm: maxBedAngle > 0 ? hoistAngle / maxBedAngle : 0,
      speed: speedReading ? numberOr(speedReading.value, 0) : 0,
      rpm: rpmReading ? numberOr(rpmReading.value, 0) : 0,
      timestamp: timestampReading ? timestampReading.value : null
    };
  }

  function parseInitArgs(telemetryOrOptions, config) {
    var telemetry = telemetryOrOptions || {};
    var settings = config || {};
    if (isObject(telemetryOrOptions) &&
        (telemetryOrOptions.telemetry !== undefined || telemetryOrOptions.config !== undefined)) {
      telemetry = telemetryOrOptions.telemetry || {};
      settings = Object.assign({}, telemetryOrOptions.config || {}, config || {});
    }
    return { telemetry: telemetry, config: settings };
  }

  function resolveContainer(container) {
    if (typeof container === 'string') {
      if (!global.document) throw new Error('A browser document is required.');
      container = global.document.querySelector(container);
    }
    if (!container || !container.appendChild) {
      throw new Error('DigitalTwinTruck.init needs a DOM container or selector.');
    }
    return container;
  }

  function loadThree(url, timeout) {
    if (global.THREE && global.THREE.Scene) {
      return Promise.resolve(global.THREE);
    }
    if (!global.document) {
      return Promise.reject(new Error('Three.js requires a browser document.'));
    }
    if (pendingThreeLoad) return pendingThreeLoad;

    pendingThreeLoad = new Promise(function (resolve, reject) {
      var settled = false;
      var script = global.document.querySelector('script[data-digital-twin-three]');
      var appendScript = !script;
      var timer = null;
      function finish(error) {
        if (settled) return;
        settled = true;
        if (timer) global.clearTimeout(timer);
        if (error) reject(error);
        else if (global.THREE && global.THREE.Scene) resolve(global.THREE);
        else reject(new Error('Three.js loaded without exposing window.THREE.'));
      }
      if (!script) {
        script = global.document.createElement('script');
        script.src = url;
        script.async = true;
        script.crossOrigin = 'anonymous';
        script.setAttribute('data-digital-twin-three', 'true');
      }
      script.addEventListener('load', function () { finish(); }, { once: true });
      script.addEventListener('error', function () {
        finish(new Error('Unable to load the Three.js CDN asset.'));
      }, { once: true });
      timer = global.setTimeout(function () {
        finish(new Error('Timed out while loading Three.js.'));
      }, numberOr(timeout, 7000));
      /* Attach listeners before insertion so a cached script cannot race us. */
      if (appendScript) {
        (global.document.head || global.document.documentElement).appendChild(script);
      }
      /* A cached script can already have finished before listeners attached. */
      if (global.THREE && global.THREE.Scene) finish();
    }).catch(function (error) {
      pendingThreeLoad = null;
      throw error;
    });
    return pendingThreeLoad;
  }

  function createCanvas() {
    var canvas = global.document.createElement('canvas');
    canvas.className = 'dt-canvas';
    canvas.setAttribute('role', 'img');
    canvas.setAttribute('aria-label', 'Dump truck digital twin visualization');
    canvas.setAttribute('data-digital-twin-canvas', 'true');
    return canvas;
  }

  function setText(element, value) {
    if (element) element.textContent = value;
  }

  function DigitalTwin(container, telemetry, config) {
    this.container = resolveContainer(container);
    this.config = mergeConfig(config);
    this._disposed = false;
    this._lastTime = (global.performance && global.performance.now)
      ? global.performance.now() : Date.now();
    this._mode = 'fallback';
    this._three = null;
    this._renderer = null;
    this._resizeObserver = null;
    this._resizeListener = null;
    this._raf = null;
    this._width = 1;
    this._height = 1;
    this._pixelRatio = 1;
    this.target = deriveTelemetry(telemetry, this.config);
    this.state = Object.assign({}, this.target);

    if (this.config.clearContainer) this.container.textContent = '';
    this.container.classList.add('dt-container');
    this.shell = global.document.createElement('div');
    this.shell.className = 'dt-canvas-shell';
    this.shell.setAttribute('data-digital-twin', 'true');
    if (this.config.height !== undefined && this.config.height !== null) {
      this.shell.style.height = typeof this.config.height === 'number'
        ? this.config.height + 'px' : String(this.config.height);
    }
    this.container.appendChild(this.shell);
    this.canvas = createCanvas();
    this.shell.appendChild(this.canvas);
    this._ctx = this.canvas.getContext('2d');
    this._createHud();
    this._installResizeHandling();
    this.resize();
    this._updateHud();

    var self = this;
    this.ready = (this.config.useThree === false
      ? Promise.reject(new Error('Three.js disabled by configuration.'))
      : (this.config.three && this.config.three.Scene
        ? Promise.resolve(this.config.three)
        : loadThree(this.config.threeUrl, this.config.threeLoadTimeout)))
      .then(function (THREE) {
        if (self._disposed) return 'disposed';
        try {
          self._switchToThree(THREE);
          return 'three';
        } catch (error) {
          self._fallbackReason = error;
          return 'fallback';
        }
      })
      .catch(function (error) {
        self._fallbackReason = error;
        return 'fallback';
      });

    this._tick = this._tick.bind(this);
    this._raf = global.requestAnimationFrame
      ? global.requestAnimationFrame(this._tick)
      : global.setTimeout(function () { self._tick(Date.now()); }, 16);
  }

  DigitalTwin.prototype._createHud = function () {
    if (this.config.showHud === false) return;
    this.hud = global.document.createElement('div');
    this.hud.className = 'dt-hud';
    this.hud.setAttribute('aria-live', 'polite');
    var definitions = [
      ['load', 'LOAD', '--'],
      ['temperature', 'TEMP', '--'],
      ['vibration', 'VIBRATION', '--'],
      ['hoist', 'HOIST', '--']
    ];
    var self = this;
    definitions.forEach(function (definition) {
      var item = global.document.createElement('div');
      item.className = 'dt-hud-item';
      var label = global.document.createElement('span');
      label.className = 'dt-hud-label';
      label.textContent = definition[1];
      var value = global.document.createElement('strong');
      value.className = 'dt-hud-value';
      value.setAttribute('data-dt-value', definition[0]);
      value.textContent = definition[2];
      item.appendChild(label);
      item.appendChild(value);
      self.hud.appendChild(item);
    });
    this.shell.appendChild(this.hud);
  };

  DigitalTwin.prototype._updateHud = function () {
    if (!this.hud) return;
    var load = this.hud.querySelector('[data-dt-value="load"]');
    var temperature = this.hud.querySelector('[data-dt-value="temperature"]');
    var vibration = this.hud.querySelector('[data-dt-value="vibration"]');
    var hoist = this.hud.querySelector('[data-dt-value="hoist"]');
    setText(load, Math.round(this.state.load * 100) + '%');
    setText(temperature, Math.round(this.state.temperature) + '°C');
    setText(vibration, Math.round(this.state.vibration * 100) + '%');
    setText(hoist, Math.round(this.state.hoistAngle) + '°');
    if (temperature) {
      temperature.style.color = temperatureColorCss(
        this.state.temperature, this.config.temperatureRange
      );
    }
  };

  DigitalTwin.prototype._installResizeHandling = function () {
    var self = this;
    if (global.ResizeObserver) {
      this._resizeObserver = new global.ResizeObserver(function () { self.resize(); });
      this._resizeObserver.observe(this.shell);
    } else if (global.addEventListener) {
      this._resizeListener = function () { self.resize(); };
      global.addEventListener('resize', this._resizeListener);
    }
  };

  DigitalTwin.prototype._measure = function () {
    var rect = this.shell.getBoundingClientRect();
    var width = rect.width || this.container.clientWidth || 900;
    var height = rect.height || numberOr(this.config.height, 460) || 460;
    this._width = Math.max(1, width);
    this._height = Math.max(1, height);
    this._pixelRatio = Math.min(
      global.devicePixelRatio || 1,
      numberOr(this.config.pixelRatioMax, 2)
    );
  };

  DigitalTwin.prototype.resize = function () {
    if (this._disposed) return this;
    this._measure();
    if (this.canvas) {
      this.canvas.width = Math.max(1, Math.round(this._width * this._pixelRatio));
      this.canvas.height = Math.max(1, Math.round(this._height * this._pixelRatio));
      this.canvas.style.width = '100%';
      this.canvas.style.height = '100%';
      if (this._ctx) this._ctx.setTransform(this._pixelRatio, 0, 0, this._pixelRatio, 0, 0);
    }
    if (this._renderer && this._camera) {
      this._renderer.setPixelRatio(this._pixelRatio);
      this._renderer.setSize(this._width, this._height, false);
      this._camera.aspect = this._width / this._height;
      this._camera.updateProjectionMatrix();
    }
    return this;
  };

  DigitalTwin.prototype._switchToThree = function (THREE) {
    var oldCanvas = this.canvas;
    var nextCanvas = createCanvas();
    this.shell.replaceChild(nextCanvas, oldCanvas);
    this.canvas = nextCanvas;
    this._ctx = null;
    try {
      this._createThreeScene(THREE);
      this._mode = 'three';
      this.resize();
    } catch (error) {
      if (this._renderer && this._renderer.dispose) this._renderer.dispose();
      this._renderer = null;
      this._three = null;
      this.shell.replaceChild(oldCanvas, nextCanvas);
      this.canvas = oldCanvas;
      this._ctx = oldCanvas.getContext('2d');
      this._mode = 'fallback';
      this.resize();
      throw error;
    }
  };

  function standardMaterial(THREE, color, options) {
    var settings = options || {};
    return new THREE.MeshStandardMaterial({
      color: color,
      roughness: settings.roughness === undefined ? 0.72 : settings.roughness,
      metalness: settings.metalness === undefined ? 0.12 : settings.metalness,
      emissive: settings.emissive || '#000000',
      emissiveIntensity: settings.emissiveIntensity || 0
    });
  }

  function addMesh(THREE, parent, geometry, material, position, name) {
    var mesh = new THREE.Mesh(geometry, material);
    if (position) mesh.position.set(position[0], position[1], position[2]);
    if (name) mesh.name = name;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    parent.add(mesh);
    return mesh;
  }

  function addBox(THREE, parent, size, material, position, name) {
    return addMesh(
      THREE, parent, new THREE.BoxGeometry(size[0], size[1], size[2]),
      material, position, name
    );
  }

  function pointOnBed(hingeX, hingeY, angle, localX, localY) {
    /* Three.js rotates the bed by -angle around z so the front rises. */
    var theta = -angle;
    return {
      x: hingeX + localX * Math.cos(theta) - localY * Math.sin(theta),
      y: hingeY + localX * Math.sin(theta) + localY * Math.cos(theta),
      z: 0
    };
  }

  function orientCylinder(THREE, mesh, a, b) {
    var start = new THREE.Vector3(a.x, a.y, a.z || 0);
    var end = new THREE.Vector3(b.x, b.y, b.z || 0);
    var direction = new THREE.Vector3().subVectors(end, start);
    var length = direction.length();
    mesh.position.copy(start).add(end).multiplyScalar(0.5);
    mesh.scale.set(1, Math.max(0.001, length / (mesh.userData.baseLength || 1)), 1);
    mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize());
  }

  DigitalTwin.prototype._createThreeScene = function (THREE) {
    if (!THREE || !THREE.Scene || !THREE.WebGLRenderer) {
      throw new Error('The supplied Three.js build is incomplete.');
    }
    this._three = THREE;
    var cfg = this.config;
    var scene = new THREE.Scene();
    scene.background = new THREE.Color(cfg.background);
    scene.fog = new THREE.Fog(cfg.background, 11, 27);
    var camera = new THREE.PerspectiveCamera(cfg.cameraFov, 1, 0.1, 100);
    camera.position.set(8.3, cfg.cameraHeight, cfg.cameraDistance);
    camera.lookAt(0, cfg.cameraTargetY, 0);

    var renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance'
    });
    renderer.setPixelRatio(this._pixelRatio);
    renderer.setSize(this._width, this._height, false);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap || renderer.shadowMap.type;
    if ('outputColorSpace' in renderer && THREE.SRGBColorSpace) {
      renderer.outputColorSpace = THREE.SRGBColorSpace;
    }
    if ('toneMapping' in renderer && THREE.ACESFilmicToneMapping) {
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 1.05;
    }

    var ambient = new THREE.HemisphereLight('#b9d8e7', '#15202a', 1.35);
    scene.add(ambient);
    var key = new THREE.DirectionalLight('#fff2d3', 2.5);
    key.position.set(-5, 9, 7);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.camera.left = -9;
    key.shadow.camera.right = 9;
    key.shadow.camera.top = 9;
    key.shadow.camera.bottom = -5;
    scene.add(key);
    var rim = new THREE.DirectionalLight('#4b8db2', 1.15);
    rim.position.set(5, 4, -7);
    scene.add(rim);

    var groundMaterial = standardMaterial(THREE, '#0b151d', { roughness: 0.94, metalness: 0 });
    var ground = addMesh(
      THREE, scene, new THREE.PlaneGeometry(34, 25), groundMaterial, [0, -0.015, 0], 'ground'
    );
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    var grid = new THREE.GridHelper(26, 26, '#1b3542', '#10212b');
    grid.position.y = 0.005;
    scene.add(grid);

    var vehicle = new THREE.Group();
    vehicle.name = 'dump-truck';
    scene.add(vehicle);
    var body = new THREE.Group();
    body.name = 'suspension-body';
    vehicle.add(body);

    var bodyMaterial = standardMaterial(THREE, cfg.bodyColor, { roughness: 0.56, metalness: 0.2 });
    var darkBodyMaterial = standardMaterial(THREE, cfg.bodyDarkColor, { roughness: 0.66, metalness: 0.18 });
    var accentMaterial = standardMaterial(THREE, cfg.accentColor, { roughness: 0.52, metalness: 0.28 });
    var glassMaterial = standardMaterial(THREE, '#102b38', {
      roughness: 0.2, metalness: 0.18, emissive: '#061822', emissiveIntensity: 0.35
    });
    var tireMaterial = standardMaterial(THREE, '#111820', { roughness: 0.92, metalness: 0.02 });
    var hubMaterial = standardMaterial(THREE, '#8b9aa1', { roughness: 0.3, metalness: 0.8 });
    var hydraulicMaterial = standardMaterial(THREE, '#c9d3d5', { roughness: 0.3, metalness: 0.75 });
    var thermalMaterial = standardMaterial(THREE, '#2c9fd0', {
      roughness: 0.32, metalness: 0.18, emissive: '#2c9fd0', emissiveIntensity: 0.7
    });

    addBox(THREE, body, [4.85, 0.30, 1.82], darkBodyMaterial, [-0.1, 1.08, 0], 'chassis');
    addBox(THREE, body, [4.1, 0.14, 1.95], bodyMaterial, [0.15, 1.27, 0], 'frame-rail');
    addBox(THREE, body, [1.58, 1.28, 1.72], bodyMaterial, [-1.78, 1.72, 0], 'cab');
    addBox(THREE, body, [0.56, 0.50, 1.74], accentMaterial, [-2.82, 1.27, 0], 'hood');
    addBox(THREE, body, [1.42, 0.14, 1.84], accentMaterial, [-1.79, 2.42, 0], 'cab-roof');
    addBox(THREE, body, [0.10, 0.55, 1.40], glassMaterial, [-2.585, 1.86, 0], 'windshield');
    addBox(THREE, body, [0.72, 0.43, 0.028], glassMaterial, [-1.86, 1.91, 0.871], 'left-window');
    addBox(THREE, body, [0.72, 0.43, 0.028], glassMaterial, [-1.86, 1.91, -0.871], 'right-window');
    addBox(THREE, body, [0.22, 0.16, 1.93], darkBodyMaterial, [-3.13, 0.98, 0], 'bumper');
    addBox(THREE, body, [0.08, 0.55, 0.10], thermalMaterial, [-1.38, 1.48, 0.91], 'thermal-bar');

    var beacon = addMesh(
      THREE, body, new THREE.SphereGeometry(0.11, 16, 10), thermalMaterial,
      [-1.78, 2.57, 0], 'thermal-beacon'
    );
    var thermalLight = new THREE.PointLight('#2c9fd0', 0.8, 3.2);
    thermalLight.position.set(-1.55, 2.0, 0.5);
    body.add(thermalLight);

    var bed = new THREE.Group();
    bed.name = 'dump-body';
    bed.position.set(1.68, 1.30, 0);
    body.add(bed);
    var bedMaterial = standardMaterial(THREE, cfg.bodyColor, { roughness: 0.62, metalness: 0.16 });
    var bedEdgeMaterial = standardMaterial(THREE, cfg.accentColor, { roughness: 0.55, metalness: 0.25 });
    addBox(THREE, bed, [2.72, 0.18, 2.24], bedMaterial, [-1.34, 0.04, 0], 'bed-floor');
    addBox(THREE, bed, [2.76, 1.02, 0.13], bedMaterial, [-1.35, 0.57, 1.08], 'bed-side-near');
    addBox(THREE, bed, [2.76, 1.02, 0.13], bedMaterial, [-1.35, 0.57, -1.08], 'bed-side-far');
    addBox(THREE, bed, [0.13, 1.04, 2.24], bedMaterial, [-2.70, 0.58, 0], 'bed-front');
    addBox(THREE, bed, [0.13, 0.94, 2.24], bedMaterial, [0.03, 0.53, 0], 'bed-tailgate');
    addBox(THREE, bed, [2.88, 0.10, 0.10], bedEdgeMaterial, [-1.35, 1.10, 1.08], 'bed-rim-near');
    addBox(THREE, bed, [2.88, 0.10, 0.10], bedEdgeMaterial, [-1.35, 1.10, -1.08], 'bed-rim-far');
    var bedInnerMaterial = standardMaterial(THREE, '#5d3a1b', { roughness: 0.9, metalness: 0.05 });
    addBox(THREE, bed, [2.45, 0.025, 1.98], bedInnerMaterial, [-1.34, 0.145, 0], 'bed-interior');

    var loadBars = [];
    for (var barIndex = 0; barIndex < 5; barIndex += 1) {
      var bar = addBox(
        THREE, bed, [0.09, 0.46, 0.035], thermalMaterial,
        [-2.22 + barIndex * 0.42, 0.43, 1.151], 'load-bar-' + barIndex
      );
      loadBars.push(bar);
    }

    var wheelPositions = [
      [-2.10, 0.50, 1.02], [-2.10, 0.50, -1.02],
      [1.30, 0.50, 1.02], [1.30, 0.50, -1.02]
    ];
    var wheels = [];
    wheelPositions.forEach(function (position, index) {
      var wheel = addMesh(
        THREE, vehicle, new THREE.CylinderGeometry(0.50, 0.50, 0.30, 24),
        tireMaterial, position, 'wheel-' + index
      );
      wheel.rotation.x = Math.PI / 2;
      var hub = addMesh(
        THREE, vehicle, new THREE.CylinderGeometry(0.19, 0.19, 0.325, 18),
        hubMaterial, position, 'hub-' + index
      );
      hub.rotation.x = Math.PI / 2;
      wheels.push({ wheel: wheel, hub: hub });
    });

    var shockMaterial = standardMaterial(THREE, '#e4a83d', { roughness: 0.4, metalness: 0.5 });
    var shocks = [];
    [[-2.10, 1.02], [-2.10, -1.02], [1.30, 1.02], [1.30, -1.02]].forEach(function (position, index) {
      var shock = addBox(
        THREE, body, [0.12, 0.42, 0.12], shockMaterial,
        [position[0], 0.82, position[1]], 'shock-' + index
      );
      shocks.push(shock);
    });

    var hydraulic = addMesh(
      THREE, body, new THREE.CylinderGeometry(0.065, 0.065, 1, 12),
      hydraulicMaterial, [0, 0, 0], 'hydraulic-cylinder'
    );
    hydraulic.userData.baseLength = 1;
    var hydraulicDark = addMesh(
      THREE, body, new THREE.CylinderGeometry(0.105, 0.105, 1, 12),
      darkBodyMaterial, [0, 0, 0], 'hydraulic-sleeve'
    );
    hydraulicDark.userData.baseLength = 1;

    var payloadPlate = addBox(
      THREE, body, [0.88, 0.035, 0.08], thermalMaterial, [0.30, 1.40, 0.93], 'payload-plate'
    );

    this._scene = scene;
    this._camera = camera;
    this._renderer = renderer;
    this._vehicle = vehicle;
    this._body = body;
    this._bed = bed;
    this._refs = {
      bodyMaterial: bodyMaterial,
      bedMaterial: bedMaterial,
      thermalMaterial: thermalMaterial,
      beacon: beacon,
      thermalLight: thermalLight,
      loadBars: loadBars,
      shocks: shocks,
      hydraulic: hydraulic,
      hydraulicDark: hydraulicDark,
      payloadPlate: payloadPlate,
      wheels: wheels,
      hingeX: 1.68,
      hingeY: 1.30
    };
  };

  DigitalTwin.prototype._updateThree = function (time) {
    var THREE = this._three;
    var refs = this._refs;
    var cfg = this.config;
    var seconds = time / 1000;
    var sag = this.state.load * numberOr(cfg.maxSuspensionSag, 0.30);
    var angle = this.state.hoistAngle * Math.PI / 180;
    var thermalColor = temperatureColorThree(THREE, this.state.temperature, cfg.temperatureRange);
    var vibration = this.state.vibration;
    var vibrationWave = Math.sin(seconds * numberOr(cfg.vibrationFrequency, 18)) +
      0.32 * Math.sin(seconds * numberOr(cfg.vibrationFrequency, 18) * 1.71 + 0.8);
    var vibrationSide = Math.sin(seconds * numberOr(cfg.vibrationFrequency, 18) * 0.67);
    var vibrationAmount = vibration * numberOr(cfg.vibrationAmplitude, 0.04);

    this._body.position.y = -sag;
    this._bed.rotation.z = -angle;
    this._vehicle.position.x = vibrationWave * vibrationAmount;
    this._vehicle.position.y = Math.abs(vibrationSide) * vibrationAmount * 0.42;
    this._vehicle.rotation.z = vibrationWave * vibrationAmount * 0.28;
    this._vehicle.rotation.x = vibrationSide * vibrationAmount * 0.18;

    refs.thermalMaterial.color.copy(thermalColor);
    refs.thermalMaterial.emissive.copy(thermalColor);
    refs.thermalMaterial.emissiveIntensity = 0.45 + this.state.temperatureNorm * 1.2;
    refs.thermalLight.color.copy(thermalColor);
    refs.thermalLight.intensity = 0.45 + this.state.temperatureNorm * 1.5;
    refs.bodyMaterial.emissive.copy(thermalColor);
    refs.bodyMaterial.emissiveIntensity = this.state.temperatureNorm * 0.08;
    refs.bedMaterial.emissive.copy(thermalColor);
    refs.bedMaterial.emissiveIntensity = this.state.temperatureNorm * 0.045;

    refs.loadBars.forEach(function (bar, index) {
      var active = this.state.load >= (index + 1) / refs.loadBars.length;
      bar.visible = active;
      bar.material.color.copy(active ? thermalColor : new THREE.Color('#26353b'));
      bar.material.emissive.copy(active ? thermalColor : new THREE.Color('#000000'));
    }, this);
    refs.payloadPlate.scale.x = 0.7 + this.state.load * 0.6;

    refs.shocks.forEach(function (shock) {
      var compression = 1 - this.state.load * 0.42;
      shock.scale.y = compression;
      shock.position.y = 0.82 - sag * 0.52;
    }, this);

    var top = pointOnBed(refs.hingeX, refs.hingeY, angle, -1.35, 0.18);
    var base = { x: 0.45, y: 1.10, z: 0 };
    orientCylinder(THREE, refs.hydraulic, base, top);
    var sleeveTop = pointOnBed(refs.hingeX, refs.hingeY, angle, -0.70, 0.13);
    orientCylinder(THREE, refs.hydraulicDark, base, sleeveTop);

    this._renderer.render(this._scene, this._camera);
  };

  DigitalTwin.prototype._drawFallback = function (time) {
    var ctx = this._ctx;
    if (!ctx) return;
    var width = this._width;
    var height = this._height;
    var cfg = this.config;
    var state = this.state;
    var seconds = time / 1000;
    var scale = Math.min(width / 920, height / 520);
    var groundY = height * 0.80;
    var bodyDrop = state.load * 25;
    var bodyY = -144 + bodyDrop;
    var bedAngle = state.hoistAngle * Math.PI / 180;
    var heat = temperatureColorCss(state.temperature, cfg.temperatureRange);
    var vibrationWave = Math.sin(seconds * numberOr(cfg.vibrationFrequency, 18)) +
      0.32 * Math.sin(seconds * numberOr(cfg.vibrationFrequency, 18) * 1.71 + 0.8);
    var vibrationAmount = state.vibration * numberOr(cfg.vibrationAmplitude, 0.04) * 420;
    var shakeX = vibrationWave * vibrationAmount;
    var shakeY = Math.abs(Math.sin(seconds * 11.5)) * vibrationAmount * 0.35;

    ctx.setTransform(this._pixelRatio, 0, 0, this._pixelRatio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    var background = ctx.createLinearGradient(0, 0, 0, height);
    background.addColorStop(0, '#071017');
    background.addColorStop(1, '#0d1b23');
    ctx.fillStyle = background;
    ctx.fillRect(0, 0, width, height);

    ctx.save();
    ctx.globalAlpha = 0.28;
    ctx.strokeStyle = '#25424e';
    ctx.lineWidth = 1;
    var gridStep = 42 * scale;
    for (var gx = 0; gx < width; gx += gridStep) {
      ctx.beginPath(); ctx.moveTo(gx, height * 0.62); ctx.lineTo(gx, height); ctx.stroke();
    }
    for (var gy = height * 0.64; gy < height; gy += gridStep * 0.62) {
      ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(width, gy); ctx.stroke();
    }
    ctx.restore();

    ctx.save();
    ctx.translate(width / 2 - 410 * scale + shakeX, groundY + shakeY);
    ctx.scale(scale, scale);
    ctx.rotate(vibrationWave * state.vibration * 0.002);

    /* Ground shadow. */
    ctx.fillStyle = 'rgba(0, 0, 0, .46)';
    ctx.beginPath();
    ctx.ellipse(420, -18, 350, 22, 0, 0, Math.PI * 2);
    ctx.fill();

    /* Wheels stay at ground level while the loaded body drops. */
    var wheelData = [[155, -55], [620, -55]];
    wheelData.forEach(function (wheel) {
      ctx.fillStyle = '#10191f';
      ctx.beginPath(); ctx.arc(wheel[0], wheel[1], 57, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = '#263840'; ctx.lineWidth = 5;
      ctx.beginPath(); ctx.arc(wheel[0], wheel[1], 44, 0, Math.PI * 2); ctx.stroke();
      ctx.fillStyle = '#a8b9bc';
      ctx.beginPath(); ctx.arc(wheel[0], wheel[1], 18, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#344b54';
      ctx.beginPath(); ctx.arc(wheel[0], wheel[1], 8, 0, Math.PI * 2); ctx.fill();
    });

    /* Suspension/shock indicators compress with the payload. */
    ctx.strokeStyle = '#e1a53f';
    ctx.lineWidth = 8;
    var shockBottom = -60;
    var shockTop = bodyY - 8;
    [[155, -1], [620, -1]].forEach(function (shock) {
      ctx.beginPath(); ctx.moveTo(shock[0], shockBottom); ctx.lineTo(shock[0], shockTop); ctx.stroke();
    });
    ctx.strokeStyle = '#f0ca73';
    ctx.lineWidth = 2;
    [[155, -1], [620, -1]].forEach(function (shock) {
      for (var coil = 0; coil < 4; coil += 1) {
        var cy = shockBottom + 10 + coil * ((shockTop - shockBottom - 20) / 4);
        ctx.beginPath();
        ctx.moveTo(shock[0] - 7, cy);
        ctx.lineTo(shock[0] + 7, cy + 5);
        ctx.stroke();
      }
    });

    /* Chassis and cab. */
    ctx.fillStyle = '#573b22';
    ctx.fillRect(80, bodyY + 45, 650, 28);
    ctx.fillStyle = cfg.bodyColor;
    ctx.fillRect(73, bodyY - 70, 190, 115);
    ctx.fillStyle = cfg.accentColor;
    ctx.fillRect(27, bodyY - 28, 58, 42);
    ctx.fillStyle = '#173441';
    ctx.fillRect(88, bodyY - 52, 76, 35);
    ctx.fillRect(174, bodyY - 52, 72, 35);
    ctx.fillStyle = '#0e222b';
    ctx.fillRect(28, bodyY - 60, 8, 48);
    ctx.fillStyle = '#d4e4e4';
    ctx.fillRect(19, bodyY + 15, 12, 9);
    ctx.fillStyle = '#27363a';
    ctx.fillRect(16, bodyY + 28, 68, 15);
    ctx.fillStyle = heat;
    ctx.fillRect(130, bodyY - 2, 8, 32);
    ctx.shadowColor = heat;
    ctx.shadowBlur = 12 + state.temperatureNorm * 12;
    ctx.fillStyle = heat;
    ctx.beginPath(); ctx.arc(172, bodyY - 82, 7, 0, Math.PI * 2); ctx.fill();
    ctx.shadowBlur = 0;

    /* Hydraulic cylinder. */
    var hingeX = 700;
    var hingeY = bodyY + 42;
    var topX = hingeX - 172 * Math.cos(bedAngle) - 2 * Math.sin(bedAngle);
    var topY = hingeY - 172 * Math.sin(bedAngle) + 2 * Math.cos(bedAngle);
    ctx.strokeStyle = '#c8d3d4';
    ctx.lineWidth = 10;
    ctx.beginPath(); ctx.moveTo(555, bodyY + 46); ctx.lineTo(topX, topY); ctx.stroke();
    ctx.strokeStyle = '#4a5960';
    ctx.lineWidth = 16;
    ctx.beginPath(); ctx.moveTo(555, bodyY + 46); ctx.lineTo(hingeX - 75 * Math.cos(bedAngle), hingeY - 75 * Math.sin(bedAngle)); ctx.stroke();

    /* Dump body rotates around the rear hinge. */
    ctx.save();
    ctx.translate(hingeX, hingeY);
    ctx.rotate(bedAngle);
    ctx.fillStyle = '#6f461f';
    ctx.beginPath(); ctx.moveTo(-292, 4); ctx.lineTo(10, 4); ctx.lineTo(7, 88); ctx.lineTo(-292, 88); ctx.closePath(); ctx.fill();
    ctx.fillStyle = cfg.bodyColor;
    ctx.beginPath(); ctx.moveTo(-288, -4); ctx.lineTo(7, -4); ctx.lineTo(5, 54); ctx.lineTo(-288, 54); ctx.closePath(); ctx.fill();
    ctx.fillStyle = cfg.bodyDarkColor;
    ctx.beginPath(); ctx.moveTo(-288, 54); ctx.lineTo(5, 54); ctx.lineTo(5, 82); ctx.lineTo(-288, 82); ctx.closePath(); ctx.fill();
    ctx.strokeStyle = cfg.accentColor;
    ctx.lineWidth = 6;
    ctx.beginPath(); ctx.moveTo(-288, -5); ctx.lineTo(7, -5); ctx.stroke();
    ctx.strokeStyle = 'rgba(245, 190, 87, .60)';
    ctx.lineWidth = 2;
    for (var rib = -245; rib < 0; rib += 52) {
      ctx.beginPath(); ctx.moveTo(rib, 4); ctx.lineTo(rib, 52); ctx.stroke();
    }
    for (var loadBar = 0; loadBar < 5; loadBar += 1) {
      if (state.load >= (loadBar + 1) / 5) {
        ctx.fillStyle = heat;
        ctx.fillRect(-242 + loadBar * 46, 29, 9, 22);
      }
    }
    ctx.restore();

    ctx.restore();

    /* A small status strip remains readable in the no-WebGL fallback. */
    ctx.fillStyle = 'rgba(8, 18, 24, .80)';
    ctx.fillRect(18, height - 39, Math.min(width - 36, 320), 22);
    ctx.fillStyle = '#a8c0c8';
    ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillText('2D FALLBACK  •  TELEMETRY LIVE', 28, height - 24);
  };

  DigitalTwin.prototype._tick = function (now) {
    if (this._disposed) return;
    var current = typeof now === 'number' ? now : Date.now();
    var delta = Math.min(0.1, Math.max(0, (current - this._lastTime) / 1000));
    this._lastTime = current;
    var smoothing = 1 - Math.exp(-7 * delta);
    var self = this;
    ['load', 'temperature', 'temperatureNorm', 'vibration', 'hoistAngle', 'hoistNorm', 'speed', 'rpm'].forEach(function (key) {
      var target = numberOr(self.target[key], self.state[key]);
      self.state[key] += (target - self.state[key]) * smoothing;
    });
    this._updateHud();
    if (this._mode === 'three' && this._renderer) {
      this._updateThree(current);
    } else {
      this._drawFallback(current);
    }
    var selfRef = this;
    this._raf = global.requestAnimationFrame
      ? global.requestAnimationFrame(this._tick)
      : global.setTimeout(function () { selfRef._tick(Date.now()); }, 16);
  };

  DigitalTwin.prototype.update = function (telemetry, config) {
    if (this._disposed) return this;
    if (config && isObject(config)) this.config = mergeConfig(Object.assign({}, this.config, config));
    this.target = deriveTelemetry(telemetry || {}, this.config);
    this._updateHud();
    return this;
  };

  /* Friendly alias for integrations that name their incoming stream explicitly. */
  DigitalTwin.prototype.updateTelemetry = DigitalTwin.prototype.update;

  DigitalTwin.prototype.getState = function () {
    return Object.assign({}, this.state, { mode: this._mode });
  };

  DigitalTwin.prototype.destroy = function () {
    if (this._disposed) return;
    this._disposed = true;
    if (this._raf !== null) {
      if (global.cancelAnimationFrame) global.cancelAnimationFrame(this._raf);
      else global.clearTimeout(this._raf);
    }
    if (this._resizeObserver) this._resizeObserver.disconnect();
    if (this._resizeListener && global.removeEventListener) {
      global.removeEventListener('resize', this._resizeListener);
    }
    if (this._renderer) this._renderer.dispose();
    if (this.shell && this.shell.parentNode) this.shell.parentNode.removeChild(this.shell);
  };

  function init(container, telemetryOrOptions, config) {
    var args = parseInitArgs(telemetryOrOptions, config);
    return new DigitalTwin(container, args.telemetry, args.config);
  }

  var publicApi = global.DigitalTwinTruck || {};
  publicApi.init = init;
  publicApi.loadThree = loadThree;
  publicApi.temperatureColor = temperatureColorCss;
  publicApi.version = '1.0.0';
  global.DigitalTwinTruck = publicApi;
  global.initDumpTruckDigitalTwin = init;
  global.initDigitalTwin = init;
  /* CommonJS is harmless in test runners, while the browser uses globals. */
  if (typeof module !== 'undefined' && module.exports) module.exports = publicApi;
}(typeof window !== 'undefined' ? window : globalThis));
