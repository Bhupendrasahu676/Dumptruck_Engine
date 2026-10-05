# Input exclusion comparison

## Interpretation

- Initial raw and sigmoid-calibrated scores are same held-trucks benchmark scores, not a newly untouched test: the prior run already tested these trucks.
- Candidates are selected by manifest.selected_model (validation policy), never by test metrics.
- A reused independent seed is a benchmark, not fresh unseen evidence for this comparison; synthetic engine variants and route families are not field validation.
- brake_wear remains a target, but without direct brake readings its classifier is unsupported; proxy scores do not establish brake condition or confirmed root cause.
- Perception LiDAR/radar inputs remain eligible; their presence in the new schema is reported below.
- RUL is not validated: no independent run-to-failure or censored histories establish remaining life.
- Guarded threshold results are separate post-hoc, second-use test evidence for human interpretation; a per-label validation alarm cap is not an overall or field false-alarm guarantee.
- Missing/null/non-finite metrics are reported as missing, never as zero. Feature counts are input dimensions, not physical sensor counts.

## Comparability

Matched original benchmark: **True**.
- source_csv_sha256: same
- split_membership: same
- calibration_truck: same
- target_columns: same

## Input features

Exclusion verified across every new input branch and subset; targets are retained.

### Old: manifest-selected dense_pca_99
- source_input_columns: 369
- model_input_columns: 256
- selected_original_attributes: 16
- eligible_names: 256
- pruned_names: 161
- pruned_names counts retained inputs after correlation pruning, not removed inputs
- LiDAR/radar eligible inputs: lidar.obstacle_confidence_pct, lidar.obstacle_distance_m, lidar.points_per_scan, lidar.scan_rate_hz, radar.object_count, radar.relative_speed_mps, radar.target_confidence_pct, radar.target_distance_m, lidar.obstacle_confidence_pct__available_mask, lidar.obstacle_distance_m__available_mask, lidar.points_per_scan__available_mask, lidar.scan_rate_hz__available_mask, radar.object_count__available_mask, radar.relative_speed_mps__available_mask, radar.target_confidence_pct__available_mask, radar.target_distance_m__available_mask, lidar.obstacle_confidence_pct__residual, lidar.obstacle_confidence_pct__rate, lidar.obstacle_confidence_pct__residual_slope_120s, lidar.obstacle_confidence_pct__mean_30s, lidar.obstacle_confidence_pct__std_30s, radar.target_confidence_pct__residual, radar.target_confidence_pct__rate, radar.target_confidence_pct__residual_slope_120s, radar.target_confidence_pct__mean_30s, radar.target_confidence_pct__std_30s

**selected_original_attributes**

- engine.oil_pressure_kpa__residual
- engine.oil_condition_pct__residual
- engine.boost_kpa__residual
- engine.coolant_flow_kg_s__residual
- coolant_condition_pct__residual
- transmission.oil_pressure_kpa__residual
- transmission.oil_condition_pct__residual
- gnss.longitude_deg
- suspension.hydraulic_oil_condition_pct__residual
- engine.oil_level_pct
- engine.coolant_temp_c__residual
- brake.stroke_pct__residual
- gnss.accuracy_m__residual
- gnss.latitude_deg
- transmission.vibration_mm_s__residual
- engine.oil_temp_c__residual

**model_input_columns**

- body_position_pct
- brake.accumulator_pressure_bar
- brake.cooling_temp_c
- brake.disc_temp_c
- brake.line_pressure_bar
- brake.oil_temp_c
- brake.pad_wear_pct
- brake.pedal_pct
- brake.stroke_pct
- chassis_pitch_deg
- coolant_condition_pct
- cumulative_fuel_energy_kwh
- cumulative_fuel_volume_l
- dss.camera_fps
- dss.distraction_score_pct
- dss.driver_presence_pct
- dss.fatigue_score_pct
- engine.air_filter_restriction_kpa
- engine.block_temp_c
- engine.block_vibration_mm_s
- engine.boost_kpa
- engine.coolant_energy_flow_kw
- engine.coolant_energy_pct
- engine.coolant_flow_kg_s
- engine.coolant_inlet_temp_c
- engine.coolant_pump_temp_c
- engine.coolant_temp_c
- engine.effective_power_kw
- engine.exhaust_energy_flow_kw
- engine.exhaust_energy_pct
- engine.exhaust_temp_c
- engine.exhaust_temperatures_c.0
- engine.exhaust_temperatures_c.1
- engine.exhaust_temperatures_c.2
- engine.exhaust_temperatures_c.3
- engine.exhaust_temperatures_c.4
- engine.exhaust_temperatures_c.5
- engine.fuel_energy_input_kw
- engine.fuel_level_pct
- engine.fuel_rate_lph
- engine.load_pct
- engine.mechanical_loss_kw
- engine.oil_condition_pct
- engine.oil_level_pct
- engine.oil_pressure_kpa
- engine.oil_temp_c
- engine.remainder_loss_kw
- engine.rpm
- engine.thermostat_open_pct
- engine.throttle_position_pct
- engine.torque_nm
- engine.transient_index_pct
- engine.vibration_rms_g
- frame.deflection_mm
- frame.fatigue_index_pct
- frame.strain_microstrain
- frame.stress_mpa
- frame.twist_deg
- gnss.accuracy_m
- gnss.altitude_m
- gnss.heading_deg
- gnss.latitude_deg
- gnss.longitude_deg
- gnss.satellites
- gnss.speed_mps
- hoist_angle_deg
- imu.acceleration_x_mps2
- imu.acceleration_y_mps2
- imu.acceleration_z_mps2
- imu.angular_rate_x_dps
- imu.angular_rate_y_dps
- imu.angular_rate_z_dps
- imu.pitch_deg
- imu.roll_deg
- imu.vibration_rms_g
- imu.yaw_deg
- lidar.obstacle_confidence_pct
- lidar.obstacle_distance_m
- lidar.points_per_scan
- lidar.scan_rate_hz
- payload.mass_t
- payload.utilization_pct
- rack_deg
- radar.object_count
- radar.relative_speed_mps
- radar.target_confidence_pct
- radar.target_distance_m
- suspension.actuator_vibration_mm_s
- suspension.compression_pct
- suspension.front_left_pressure_bar
- suspension.front_right_pressure_bar
- suspension.hydraulic_oil_condition_pct
- suspension.hydraulic_pressure_bar
- suspension.left_load_pct
- suspension.rear_left_pressure_bar
- suspension.rear_right_pressure_bar
- suspension.ride_height_mm
- suspension.right_load_pct
- suspension.shock_temp_c
- transmission.axle_temp_c
- transmission.efficiency_pct
- transmission.gear
- transmission.oil_condition_pct
- transmission.oil_pressure_kpa
- transmission.oil_temp_c
- transmission.output_rpm
- transmission.output_torque_nm
- transmission.powertrain_oil_pressure_bar
- transmission.slip_pct
- transmission.vibration_mm_s
- brake.disc_temp_c__available_mask
- coolant_condition_pct__available_mask
- engine.block_vibration_mm_s__available_mask
- engine.boost_kpa__available_mask
- engine.coolant_flow_kg_s__available_mask
- engine.coolant_temp_c__available_mask
- engine.oil_condition_pct__available_mask
- engine.oil_pressure_kpa__available_mask
- suspension.hydraulic_oil_condition_pct__available_mask
- transmission.oil_pressure_kpa__available_mask
- transmission.oil_temp_c__available_mask
- context.ambient_c
- context.transient
- context.cold_start
- context.mode_idle
- context.mode_hauling
- context.mode_downslope
- engine.block_temp_c__residual
- engine.block_temp_c__rate
- engine.block_temp_c__residual_slope_120s
- engine.block_temp_c__mean_30s
- engine.block_temp_c__std_30s
- engine.coolant_temp_c__residual
- engine.coolant_temp_c__rate
- engine.coolant_temp_c__residual_slope_120s
- engine.coolant_temp_c__mean_30s
- engine.coolant_temp_c__std_30s
- engine.oil_temp_c__residual
- engine.oil_temp_c__rate
- engine.oil_temp_c__residual_slope_120s
- engine.oil_temp_c__mean_30s
- engine.oil_temp_c__std_30s
- engine.exhaust_temp_c__residual
- engine.exhaust_temp_c__rate
- engine.exhaust_temp_c__residual_slope_120s
- engine.exhaust_temp_c__mean_30s
- engine.exhaust_temp_c__std_30s
- engine.coolant_pump_temp_c__residual
- engine.coolant_pump_temp_c__rate
- engine.coolant_pump_temp_c__residual_slope_120s
- engine.coolant_pump_temp_c__mean_30s
- engine.coolant_pump_temp_c__std_30s
- engine.oil_pressure_kpa__residual
- engine.oil_pressure_kpa__rate
- engine.oil_pressure_kpa__residual_slope_120s
- engine.oil_pressure_kpa__mean_30s
- engine.oil_pressure_kpa__std_30s
- engine.block_vibration_mm_s__residual
- engine.block_vibration_mm_s__rate
- engine.block_vibration_mm_s__residual_slope_120s
- engine.block_vibration_mm_s__mean_30s
- engine.block_vibration_mm_s__std_30s
- engine.oil_condition_pct__residual
- engine.oil_condition_pct__rate
- engine.oil_condition_pct__residual_slope_120s
- engine.oil_condition_pct__mean_30s
- engine.oil_condition_pct__std_30s
- engine.oil_level_pct__residual
- engine.oil_level_pct__rate
- engine.oil_level_pct__residual_slope_120s
- engine.oil_level_pct__mean_30s
- engine.oil_level_pct__std_30s
- engine.coolant_flow_kg_s__residual
- engine.coolant_flow_kg_s__rate
- engine.coolant_flow_kg_s__residual_slope_120s
- engine.coolant_flow_kg_s__mean_30s
- engine.coolant_flow_kg_s__std_30s
- engine.thermostat_open_pct__residual
- engine.thermostat_open_pct__rate
- engine.thermostat_open_pct__residual_slope_120s
- engine.thermostat_open_pct__mean_30s
- engine.thermostat_open_pct__std_30s
- engine.boost_kpa__residual
- engine.boost_kpa__rate
- engine.boost_kpa__residual_slope_120s
- engine.boost_kpa__mean_30s
- engine.boost_kpa__std_30s
- transmission.oil_temp_c__residual
- transmission.oil_temp_c__rate
- transmission.oil_temp_c__residual_slope_120s
- transmission.oil_temp_c__mean_30s
- transmission.oil_temp_c__std_30s
- transmission.oil_pressure_kpa__residual
- transmission.oil_pressure_kpa__rate
- transmission.oil_pressure_kpa__residual_slope_120s
- transmission.oil_pressure_kpa__mean_30s
- transmission.oil_pressure_kpa__std_30s
- transmission.vibration_mm_s__residual
- transmission.vibration_mm_s__rate
- transmission.vibration_mm_s__residual_slope_120s
- transmission.vibration_mm_s__mean_30s
- transmission.vibration_mm_s__std_30s
- transmission.oil_condition_pct__residual
- transmission.oil_condition_pct__rate
- transmission.oil_condition_pct__residual_slope_120s
- transmission.oil_condition_pct__mean_30s
- transmission.oil_condition_pct__std_30s
- suspension.hydraulic_oil_condition_pct__residual
- suspension.hydraulic_oil_condition_pct__rate
- suspension.hydraulic_oil_condition_pct__residual_slope_120s
- suspension.hydraulic_oil_condition_pct__mean_30s
- suspension.hydraulic_oil_condition_pct__std_30s
- suspension.actuator_vibration_mm_s__residual
- suspension.actuator_vibration_mm_s__rate
- suspension.actuator_vibration_mm_s__residual_slope_120s
- suspension.actuator_vibration_mm_s__mean_30s
- suspension.actuator_vibration_mm_s__std_30s
- coolant_condition_pct__residual
- coolant_condition_pct__rate
- coolant_condition_pct__residual_slope_120s
- coolant_condition_pct__mean_30s
- coolant_condition_pct__std_30s
- brake.pad_wear_pct__residual
- brake.pad_wear_pct__rate
- brake.pad_wear_pct__residual_slope_120s
- brake.pad_wear_pct__mean_30s
- brake.pad_wear_pct__std_30s
- brake.stroke_pct__residual
- brake.stroke_pct__rate
- brake.stroke_pct__residual_slope_120s
- brake.stroke_pct__mean_30s
- brake.stroke_pct__std_30s
- brake.disc_temp_c__residual
- brake.disc_temp_c__rate
- brake.disc_temp_c__residual_slope_120s
- brake.disc_temp_c__mean_30s
- brake.disc_temp_c__std_30s
- gnss.accuracy_m__residual
- gnss.accuracy_m__rate
- gnss.accuracy_m__residual_slope_120s
- gnss.accuracy_m__mean_30s
- gnss.accuracy_m__std_30s
- lidar.obstacle_confidence_pct__residual
- lidar.obstacle_confidence_pct__rate
- lidar.obstacle_confidence_pct__residual_slope_120s
- lidar.obstacle_confidence_pct__mean_30s
- lidar.obstacle_confidence_pct__std_30s
- radar.target_confidence_pct__residual
- radar.target_confidence_pct__rate
- radar.target_confidence_pct__residual_slope_120s
- radar.target_confidence_pct__mean_30s
- radar.target_confidence_pct__std_30s
- engine.egt_spread_c
- leadlag.engine.coolant_flow_kg_s__to__engine.oil_temp_c_s
- leadlag.engine.oil_pressure_kpa__to__engine.oil_temp_c_s
- leadlag.engine.oil_temp_c__to__engine.block_vibration_mm_s_s
### New: manifest-selected dense_pca_99
- source_input_columns: 318
- model_input_columns: 220
- selected_original_attributes: 32
- eligible_names: 220
- pruned_names: 138
- pruned_names counts retained inputs after correlation pruning, not removed inputs
- LiDAR/radar eligible inputs: lidar.obstacle_confidence_pct, lidar.obstacle_distance_m, lidar.points_per_scan, lidar.scan_rate_hz, radar.object_count, radar.relative_speed_mps, radar.target_confidence_pct, radar.target_distance_m, lidar.obstacle_confidence_pct__available_mask, lidar.obstacle_distance_m__available_mask, lidar.points_per_scan__available_mask, lidar.scan_rate_hz__available_mask, radar.object_count__available_mask, radar.relative_speed_mps__available_mask, radar.target_confidence_pct__available_mask, radar.target_distance_m__available_mask, lidar.obstacle_confidence_pct__residual, lidar.obstacle_confidence_pct__rate, lidar.obstacle_confidence_pct__residual_slope_120s, lidar.obstacle_confidence_pct__mean_30s, lidar.obstacle_confidence_pct__std_30s, radar.target_confidence_pct__residual, radar.target_confidence_pct__rate, radar.target_confidence_pct__residual_slope_120s, radar.target_confidence_pct__mean_30s, radar.target_confidence_pct__std_30s

**selected_original_attributes**

- engine.oil_pressure_kpa__residual
- engine.oil_condition_pct__residual
- engine.boost_kpa__residual
- engine.coolant_flow_kg_s__residual
- coolant_condition_pct__residual
- transmission.oil_pressure_kpa__residual
- transmission.oil_condition_pct__residual
- cumulative_fuel_volume_l
- suspension.hydraulic_oil_condition_pct__residual
- engine.oil_condition_pct__mean_30s
- engine.oil_level_pct__mean_30s
- engine.coolant_temp_c__residual
- lidar.obstacle_confidence_pct__mean_30s
- transmission.oil_pressure_kpa__available_mask
- transmission.vibration_mm_s__residual
- engine.oil_temp_c__residual
- engine.oil_temp_c__mean_30s
- engine.block_vibration_mm_s__mean_30s
- engine.block_temp_c__mean_30s
- leadlag.engine.oil_temp_c__to__engine.block_vibration_mm_s_s
- engine.oil_pressure_kpa__mean_30s
- engine.exhaust_temp_c__residual
- engine.exhaust_temp_c__mean_30s
- engine.vibration_rms_g
- engine.coolant_pump_temp_c__residual
- engine.coolant_temp_c
- engine.coolant_flow_kg_s__mean_30s
- engine.block_vibration_mm_s__residual
- context.transient
- leadlag.engine.oil_pressure_kpa__to__engine.oil_temp_c_s
- engine.coolant_inlet_temp_c
- transmission.vibration_mm_s__mean_30s

**model_input_columns**

- body_position_pct
- chassis_pitch_deg
- coolant_condition_pct
- cumulative_fuel_energy_kwh
- cumulative_fuel_volume_l
- dss.camera_fps
- dss.distraction_score_pct
- dss.driver_presence_pct
- dss.fatigue_score_pct
- engine.air_filter_restriction_kpa
- engine.block_temp_c
- engine.block_vibration_mm_s
- engine.boost_kpa
- engine.coolant_energy_flow_kw
- engine.coolant_energy_pct
- engine.coolant_flow_kg_s
- engine.coolant_inlet_temp_c
- engine.coolant_pump_temp_c
- engine.coolant_temp_c
- engine.effective_power_kw
- engine.exhaust_energy_flow_kw
- engine.exhaust_energy_pct
- engine.exhaust_temp_c
- engine.exhaust_temperatures_c.0
- engine.exhaust_temperatures_c.1
- engine.exhaust_temperatures_c.2
- engine.exhaust_temperatures_c.3
- engine.exhaust_temperatures_c.4
- engine.exhaust_temperatures_c.5
- engine.fuel_energy_input_kw
- engine.fuel_level_pct
- engine.fuel_rate_lph
- engine.load_pct
- engine.mechanical_loss_kw
- engine.oil_condition_pct
- engine.oil_level_pct
- engine.oil_pressure_kpa
- engine.oil_temp_c
- engine.remainder_loss_kw
- engine.rpm
- engine.thermostat_open_pct
- engine.throttle_position_pct
- engine.torque_nm
- engine.transient_index_pct
- engine.vibration_rms_g
- frame.deflection_mm
- frame.fatigue_index_pct
- frame.strain_microstrain
- frame.stress_mpa
- frame.twist_deg
- hoist_angle_deg
- imu.acceleration_x_mps2
- imu.acceleration_y_mps2
- imu.acceleration_z_mps2
- imu.angular_rate_x_dps
- imu.angular_rate_y_dps
- imu.angular_rate_z_dps
- imu.pitch_deg
- imu.roll_deg
- imu.vibration_rms_g
- imu.yaw_deg
- lidar.obstacle_confidence_pct
- lidar.obstacle_distance_m
- lidar.points_per_scan
- lidar.scan_rate_hz
- payload.mass_t
- payload.utilization_pct
- rack_deg
- radar.object_count
- radar.relative_speed_mps
- radar.target_confidence_pct
- radar.target_distance_m
- suspension.actuator_vibration_mm_s
- suspension.compression_pct
- suspension.front_left_pressure_bar
- suspension.front_right_pressure_bar
- suspension.hydraulic_oil_condition_pct
- suspension.hydraulic_pressure_bar
- suspension.left_load_pct
- suspension.rear_left_pressure_bar
- suspension.rear_right_pressure_bar
- suspension.ride_height_mm
- suspension.right_load_pct
- suspension.shock_temp_c
- transmission.axle_temp_c
- transmission.efficiency_pct
- transmission.gear
- transmission.oil_condition_pct
- transmission.oil_pressure_kpa
- transmission.oil_temp_c
- transmission.output_rpm
- transmission.output_torque_nm
- transmission.powertrain_oil_pressure_bar
- transmission.slip_pct
- transmission.vibration_mm_s
- coolant_condition_pct__available_mask
- engine.block_vibration_mm_s__available_mask
- engine.boost_kpa__available_mask
- engine.coolant_flow_kg_s__available_mask
- engine.coolant_temp_c__available_mask
- engine.oil_condition_pct__available_mask
- engine.oil_pressure_kpa__available_mask
- suspension.hydraulic_oil_condition_pct__available_mask
- transmission.oil_pressure_kpa__available_mask
- transmission.oil_temp_c__available_mask
- context.ambient_c
- context.transient
- context.cold_start
- context.mode_idle
- context.mode_hauling
- context.mode_downslope
- engine.block_temp_c__residual
- engine.block_temp_c__rate
- engine.block_temp_c__residual_slope_120s
- engine.block_temp_c__mean_30s
- engine.block_temp_c__std_30s
- engine.coolant_temp_c__residual
- engine.coolant_temp_c__rate
- engine.coolant_temp_c__residual_slope_120s
- engine.coolant_temp_c__mean_30s
- engine.coolant_temp_c__std_30s
- engine.oil_temp_c__residual
- engine.oil_temp_c__rate
- engine.oil_temp_c__residual_slope_120s
- engine.oil_temp_c__mean_30s
- engine.oil_temp_c__std_30s
- engine.exhaust_temp_c__residual
- engine.exhaust_temp_c__rate
- engine.exhaust_temp_c__residual_slope_120s
- engine.exhaust_temp_c__mean_30s
- engine.exhaust_temp_c__std_30s
- engine.coolant_pump_temp_c__residual
- engine.coolant_pump_temp_c__rate
- engine.coolant_pump_temp_c__residual_slope_120s
- engine.coolant_pump_temp_c__mean_30s
- engine.coolant_pump_temp_c__std_30s
- engine.oil_pressure_kpa__residual
- engine.oil_pressure_kpa__rate
- engine.oil_pressure_kpa__residual_slope_120s
- engine.oil_pressure_kpa__mean_30s
- engine.oil_pressure_kpa__std_30s
- engine.block_vibration_mm_s__residual
- engine.block_vibration_mm_s__rate
- engine.block_vibration_mm_s__residual_slope_120s
- engine.block_vibration_mm_s__mean_30s
- engine.block_vibration_mm_s__std_30s
- engine.oil_condition_pct__residual
- engine.oil_condition_pct__rate
- engine.oil_condition_pct__residual_slope_120s
- engine.oil_condition_pct__mean_30s
- engine.oil_condition_pct__std_30s
- engine.oil_level_pct__residual
- engine.oil_level_pct__rate
- engine.oil_level_pct__residual_slope_120s
- engine.oil_level_pct__mean_30s
- engine.oil_level_pct__std_30s
- engine.coolant_flow_kg_s__residual
- engine.coolant_flow_kg_s__rate
- engine.coolant_flow_kg_s__residual_slope_120s
- engine.coolant_flow_kg_s__mean_30s
- engine.coolant_flow_kg_s__std_30s
- engine.thermostat_open_pct__residual
- engine.thermostat_open_pct__rate
- engine.thermostat_open_pct__residual_slope_120s
- engine.thermostat_open_pct__mean_30s
- engine.thermostat_open_pct__std_30s
- engine.boost_kpa__residual
- engine.boost_kpa__rate
- engine.boost_kpa__residual_slope_120s
- engine.boost_kpa__mean_30s
- engine.boost_kpa__std_30s
- transmission.oil_temp_c__residual
- transmission.oil_temp_c__rate
- transmission.oil_temp_c__residual_slope_120s
- transmission.oil_temp_c__mean_30s
- transmission.oil_temp_c__std_30s
- transmission.oil_pressure_kpa__residual
- transmission.oil_pressure_kpa__rate
- transmission.oil_pressure_kpa__residual_slope_120s
- transmission.oil_pressure_kpa__mean_30s
- transmission.oil_pressure_kpa__std_30s
- transmission.vibration_mm_s__residual
- transmission.vibration_mm_s__rate
- transmission.vibration_mm_s__residual_slope_120s
- transmission.vibration_mm_s__mean_30s
- transmission.vibration_mm_s__std_30s
- transmission.oil_condition_pct__residual
- transmission.oil_condition_pct__rate
- transmission.oil_condition_pct__residual_slope_120s
- transmission.oil_condition_pct__mean_30s
- transmission.oil_condition_pct__std_30s
- suspension.hydraulic_oil_condition_pct__residual
- suspension.hydraulic_oil_condition_pct__rate
- suspension.hydraulic_oil_condition_pct__residual_slope_120s
- suspension.hydraulic_oil_condition_pct__mean_30s
- suspension.hydraulic_oil_condition_pct__std_30s
- suspension.actuator_vibration_mm_s__residual
- suspension.actuator_vibration_mm_s__rate
- suspension.actuator_vibration_mm_s__residual_slope_120s
- suspension.actuator_vibration_mm_s__mean_30s
- suspension.actuator_vibration_mm_s__std_30s
- coolant_condition_pct__residual
- coolant_condition_pct__rate
- coolant_condition_pct__residual_slope_120s
- coolant_condition_pct__mean_30s
- coolant_condition_pct__std_30s
- lidar.obstacle_confidence_pct__residual
- lidar.obstacle_confidence_pct__rate
- lidar.obstacle_confidence_pct__residual_slope_120s
- lidar.obstacle_confidence_pct__mean_30s
- lidar.obstacle_confidence_pct__std_30s
- radar.target_confidence_pct__residual
- radar.target_confidence_pct__rate
- radar.target_confidence_pct__residual_slope_120s
- radar.target_confidence_pct__mean_30s
- radar.target_confidence_pct__std_30s
- engine.egt_spread_c
- leadlag.engine.coolant_flow_kg_s__to__engine.oil_temp_c_s
- leadlag.engine.oil_pressure_kpa__to__engine.oil_temp_c_s
- leadlag.engine.oil_temp_c__to__engine.block_vibration_mm_s_s

## Initial selected-model test scores

Same held-trucks, prior-tested benchmark. Deltas withheld if not comparable.

### raw: dense_pca_99_raw → dense_pca_99_raw

| Metric | Old | New | New − old |
| --- | ---: | ---: | ---: |
| macro_AP | 0.890657 | 0.832605 | -0.0580524 |
| micro_AP | 0.915841 | 0.872437 | -0.0434049 |
| macro_F1 | 0.846196 | 0.788023 | -0.0581722 |
| macro_recall | 0.849075 | 0.813457 | -0.0356181 |
| micro_F1 | 0.810255 | 0.671894 | -0.138362 |
| hamming_loss | 0.0213765 | 0.0428063 | 0.0214298 |
| exact_match | 0.724659 | 0.545322 | -0.179337 |
| normal_false_alarm_rate | 0.264136 | 0.468253 | 0.204117 |
| macro_Brier | 0.02226 | 0.0336973 | 0.0114373 |

### sigmoid_calibrated: dense_pca_99_sigmoid_calibrated → dense_pca_99_sigmoid_calibrated

| Metric | Old | New | New − old |
| --- | ---: | ---: | ---: |
| macro_AP | 0.890609 | 0.832575 | -0.0580336 |
| micro_AP | 0.913299 | 0.875097 | -0.0382017 |
| macro_F1 | 0.84078 | 0.781226 | -0.059554 |
| macro_recall | 0.892493 | 0.762087 | -0.130406 |
| micro_F1 | 0.570013 | 0.81418 | 0.244167 |
| hamming_loss | 0.0722517 | 0.0187723 | -0.0534793 |
| exact_match | 0.0344786 | 0.748091 | 0.713613 |
| normal_false_alarm_rate | 1 | 0.0601524 | -0.939848 |
| macro_Brier | 0.0119814 | 0.014451 | 0.00246956 |

## Per-fault initial test scores

Weak flags: AP, F1 or recall < 0.5 in either run; reporting only, never model selection.

| Phase | Fault | Old AP | New AP | Old F1 | New F1 | Old recall | New recall | Weak / unsupported |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw | fault_label.engine_oil_pump_wear | 0.898849 | 0.877651 | 0.841571 | 0.807115 | 0.766051 | 0.706226 | none observed |
| raw | fault_label.piston_ring_damage | 0.989714 | 0.987095 | 0.95423 | 0.949188 | 0.95755 | 0.955463 | none observed |
| raw | fault_label.turbo_exhaust_fault | 0.957905 | 0.965424 | 0.936392 | 0.932267 | 0.912602 | 0.888211 | none observed |
| raw | fault_label.coolant_system_failure | 0.995018 | 0.98433 | 0.961155 | 0.919397 | 0.938716 | 0.859922 | none observed |
| raw | fault_label.coolant_degradation | 0.984144 | 0.989421 | 0.96738 | 0.95211 | 0.938211 | 0.909046 | none observed |
| raw | fault_label.gearbox_seal_wear | 0.889486 | 0.90097 | 0.797421 | 0.78035 | 0.800954 | 0.850034 | none observed |
| raw | fault_label.gearbox_bearing_wear | 0.991068 | 0.982466 | 0.957265 | 0.951561 | 0.923711 | 0.91134 | none observed |
| raw | fault_label.hydraulic_normal_aging | 0.118033 | 0.0885491 | 0.194431 | 0.142119 | 0.304754 | 0.300078 | old AP<0.5, old F1<0.5, old recall<0.5, new AP<0.5, new F1<0.5, new recall<0.5 |
| raw | fault_label.hydraulic_contamination | 0.995673 | 0.996201 | 0.978235 | 0.985265 | 0.984076 | 0.984873 | none observed |
| raw | fault_label.bearing_failure | 0.952991 | 0.953786 | 0.901961 | 0.898961 | 0.874143 | 0.88972 | none observed |
| raw | fault_label.low_oil | 0.991838 | 0.995785 | 0.971654 | 0.970405 | 0.975494 | 0.98498 | none observed |
| raw | fault_label.overheat | 0.946293 | 0.948567 | 0.915235 | 0.909703 | 0.930151 | 0.944121 | none observed |
| raw | fault_label.brake_wear | 0.949106 | 0.062065 | 0.864789 | 0.11539 | 0.895044 | 0.420797 | new AP<0.5, new F1<0.5, new recall<0.5, unsupported without direct brake readings |
| raw | fault_label.perception_degraded | 0.996581 | 0.991093 | 0.967033 | 0.970336 | 0.990728 | 0.974834 | none observed |
| raw | fault_label.sensor_fault | 0.745601 | 0.722786 | 0.590994 | 0.544334 | 0.734266 | 0.699767 | none observed |
| raw | fault_label.unrelated_multi_blip | 0.848214 | 0.875488 | 0.739384 | 0.779874 | 0.658754 | 0.735905 | none observed |
| sigmoid_calibrated | fault_label.engine_oil_pump_wear | 0.898811 | 0.877628 | 0.842302 | 0.807115 | 0.768969 | 0.706226 | none observed |
| sigmoid_calibrated | fault_label.piston_ring_damage | 0.989706 | 0.987095 | 0.953899 | 0.949188 | 0.95755 | 0.955463 | none observed |
| sigmoid_calibrated | fault_label.turbo_exhaust_fault | 0.957902 | 0.965423 | 0.936392 | 0.932267 | 0.912602 | 0.888211 | none observed |
| sigmoid_calibrated | fault_label.coolant_system_failure | 0.995018 | 0.98433 | 0.962705 | 0.919397 | 0.941634 | 0.859922 | none observed |
| sigmoid_calibrated | fault_label.coolant_degradation | 0.984119 | 0.989421 | 0.967364 | 0.95211 | 0.937716 | 0.909046 | none observed |
| sigmoid_calibrated | fault_label.gearbox_seal_wear | 0.889486 | 0.90097 | 0.798774 | 0.78035 | 0.799591 | 0.850034 | none observed |
| sigmoid_calibrated | fault_label.gearbox_bearing_wear | 0.991068 | 0.982457 | 0.957234 | 0.951561 | 0.923024 | 0.91134 | none observed |
| sigmoid_calibrated | fault_label.hydraulic_normal_aging | 0.117922 | 0.0885044 | 0.0990466 | 0.0564263 | 1 | 0.035074 | old AP<0.5, old F1<0.5, new AP<0.5, new F1<0.5, new recall<0.5 |
| sigmoid_calibrated | fault_label.hydraulic_contamination | 0.995673 | 0.996201 | 0.97901 | 0.985265 | 0.984076 | 0.984873 | none observed |
| sigmoid_calibrated | fault_label.bearing_failure | 0.952403 | 0.953392 | 0.903081 | 0.898961 | 0.876636 | 0.88972 | none observed |
| sigmoid_calibrated | fault_label.low_oil | 0.991838 | 0.995785 | 0.972058 | 0.970405 | 0.976285 | 0.98498 | none observed |
| sigmoid_calibrated | fault_label.overheat | 0.946293 | 0.948567 | 0.914613 | 0.909703 | 0.928987 | 0.944121 | none observed |
| sigmoid_calibrated | fault_label.brake_wear | 0.949106 | 0.0620656 | 0.864789 | 0 | 0.895044 | 0 | new AP<0.5, new F1<0.5, new recall<0.5, unsupported without direct brake readings |
| sigmoid_calibrated | fault_label.perception_degraded | 0.996581 | 0.991093 | 0.967658 | 0.970336 | 0.990728 | 0.974834 | none observed |
| sigmoid_calibrated | fault_label.sensor_fault | 0.745601 | 0.722786 | 0.593114 | 0.636651 | 0.726807 | 0.563636 | none observed |
| sigmoid_calibrated | fault_label.unrelated_multi_blip | 0.848214 | 0.875488 | 0.740433 | 0.779874 | 0.660237 | 0.735905 | none observed |

## Optional independent-seed benchmark

different or missing seeds: no matched independent-seed benchmark established
Matched benchmark: **False**.
- seed: different
- target_columns: same
- independent_engine_variant_seeds: different
- rows: same
- batch_csv_sha256: different

| Metric | Old | New | New − old |
| --- | ---: | ---: | ---: |
| macro_AP | 0.889711 | 0.830672 | missing |
| micro_AP | 0.88418 | 0.868944 | missing |
| macro_F1 | 0.857427 | 0.797811 | missing |
| macro_recall | 0.823256 | 0.745027 | missing |
| micro_F1 | 0.841855 | 0.814793 | missing |
| hamming_loss | 0.01675 | 0.0196622 | missing |
| exact_match | 0.760863 | 0.723826 | missing |
| normal_false_alarm_rate | 0.0295608 | 0.046054 | missing |
| macro_Brier | 0.014489 | 0.0168366 | missing |

## Optional guarded thresholds (separate evidence)

post-hoc second-use, separate from initial test; human interpretation only

### exact_validation_max_F1 / test_second_use_exploratory

| Metric | Old | New | New − old |
| --- | ---: | ---: | ---: |
| macro_AP | 0.890609 | missing | missing |
| micro_AP | 0.913299 | missing | missing |
| macro_F1 | 0.846411 | missing | missing |
| macro_recall | 0.847783 | missing | missing |
| micro_F1 | 0.810673 | missing | missing |
| hamming_loss | 0.0212877 | missing | missing |
| exact_match | 0.725349 | missing | missing |
| normal_false_alarm_rate | 0.261061 | missing | missing |
| macro_Brier | 0.0119814 | missing | missing |

### exact_validation_max_F1 / validation_tuning

| Metric | Old | New | New − old |
| --- | ---: | ---: | ---: |
| macro_AP | 0.895145 | missing | missing |
| micro_AP | 0.910054 | missing | missing |
| macro_F1 | 0.877346 | missing | missing |
| macro_recall | 0.875626 | missing | missing |
| micro_F1 | 0.842175 | missing | missing |
| hamming_loss | 0.0173421 | missing | missing |
| exact_match | 0.760295 | missing | missing |
| normal_false_alarm_rate | 0.22708 | missing | missing |
| macro_Brier | 0.012424 | missing | missing |

### normal_guard_1pct_per_fault / test_second_use_exploratory

| Metric | Old | New | New − old |
| --- | ---: | ---: | ---: |
| macro_AP | 0.890609 | missing | missing |
| micro_AP | 0.913299 | missing | missing |
| macro_F1 | 0.83931 | missing | missing |
| macro_recall | 0.819599 | missing | missing |
| micro_F1 | 0.853513 | missing | missing |
| hamming_loss | 0.0150133 | missing | missing |
| exact_match | 0.794144 | missing | missing |
| normal_false_alarm_rate | 0.0696431 | missing | missing |
| macro_Brier | 0.0119814 | missing | missing |

### normal_guard_1pct_per_fault / validation_tuning

| Metric | Old | New | New − old |
| --- | ---: | ---: | ---: |
| macro_AP | 0.895145 | missing | missing |
| micro_AP | 0.910054 | missing | missing |
| macro_F1 | 0.856239 | missing | missing |
| macro_recall | 0.833235 | missing | missing |
| micro_F1 | 0.869799 | missing | missing |
| hamming_loss | 0.013116 | missing | missing |
| exact_match | 0.807383 | missing | missing |
| normal_false_alarm_rate | 0.0501923 | missing | missing |
| macro_Brier | 0.012424 | missing | missing |

Full saved candidate and per-fault comparisons, including missing fields, are in the companion JSON.
