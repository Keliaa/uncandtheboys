The RETROFIT AI dataset contains simulated sensor measurements from 12 pumps, with seven days of data per pump.
Most values are numerical. The files also contain times, categories, maintenance notes, and separate fault labels.
The goal is to detect developing machine or sensor problems, explain the evidence, and recommend a human check.
The assistant gives advice only. It does not control a pump.

Think of the measurements as a diary and the fault labels as an answer sheet.
Each measurement row describes one pump at one time. A fault label describes a problem over a period of time.
One event can cover thousands of rows. Events can overlap.

The overview below comes first. Compact tables then explain every field in the main data tables and the additional measurement files.
The full types, permitted ranges, and missing-value rules are in [DATA_DICTIONARY.md](../../../DATA_DICTIONARY.md).

The available versions have different sizes and time intervals.

| Version | Contents | Main files |
|---|---|---|
| Full | 12 pumps, seven days each, one-second row grid, about 7.2 million measurement rows. | `data/full/telemetry/`, stored as Parquet tables. |
| Starter | Five of the same pumps, resampled to ten-second rows, about 301,000 rows. | `data/starter/telemetry.csv.gz`, a compressed CSV table. |
| One-minute summaries | Average, spread, minimum, maximum, and sample count for each signal. | `data/full/features_1min.parquet`. |
| Vibration recordings | One-second recordings at 10,240 samples per second, every four hours while running. Available on T1 and T2 pumps. | `data/full/vibration_snapshots.parquet`. |
| Optional stress data | Newly generated pumps with different conditions and missing sensors. Only about 30% of events are labelled. | Generate separately with `make_dataset.py --tier stress`. |

CSV and Parquet both store tables. Parquet stores large numerical tables efficiently. The vibration recordings contain lists of numbers, not images.

**`features_1min.parquet` contains prepared data. It is not a trained model or a test model.**
The name means sensor **features**, summarised over **one minute**, stored in the **Parquet** table format.
A feature is a value that a detection method can use, such as average current during a minute.
The file contains summaries for training, validation, and test pumps. Select the correct rows using the split in `assets.csv`.
The detailed explanation and field list appear below.

The full dataset is split by pump. A pump belongs to only one split.

| Split | Pumps | Purpose | Fault labels |
|---|---|---|---|
| Training | PUMP-01 to PUMP-07 | Fit the model and its initial settings. | Included in `events_train.csv`. |
| Validation | PUMP-08 and PUMP-09 | Compare models and select improvements. | Included in `events_validation.csv`. |
| Test | PUMP-10 to PUMP-12 | Produce final warnings for organiser scoring. | Hidden. Measurements are public. |

The starter subset contains training pumps PUMP-01, PUMP-03, PUMP-04, and PUMP-06, plus validation pump PUMP-08. It contains no test pumps.
The data windows of different splits do not overlap in time. Test includes a new vessel and a new plant.
Keep the supplied splits. Do not randomly mix measurement rows from these pumps into new splits.

Use training data for fitting and validation data for model selection. Do not tune settings or manually assign labels using test measurements.
An alert at time `t` can use only information available at or before `t`, plus static pump information.
Fault labels, later repairs, and future measurements must not become detector inputs.
Use a complete one-minute summary only after that minute has ended.

Not every pump has the same sensors.

| Sensor level | Available measurements |
|---|---|
| `T0_LEGACY` | Current, discharge pressure, liquid temperature, and room temperature. Drive speed and power are available where fitted. |
| `T1_RETROFIT` | T0 plus suction pressure and wireless vibration and bearing temperature. Wireless measurements arrive about every five minutes. |
| `T2_FULL` | More frequent measurements and more instruments, including flow, power, and motor winding temperature. Valve position is available where applicable. |

Check `assets.csv:channels_available` for each pump. A blank value can mean no sensor, no update at that time, or a lost sample.
Missing values are not zero. Gateway outages can remove entire rows. `RUNNING` describes operation, not machine health.
All times are UTC. Pressure uses gauge units: zero means atmospheric pressure. Negative suction pressure can be valid.
In this simulation, absolute pressure equals gauge pressure plus 101.3 kPa. Fault growth is compressed into seven days.

The telemetry table contains the sensor readings. Full data are grouped by pump and day under `data/full/telemetry/`.

| Field | Meaning |
|---|---|
| `asset_id` | Pump identifier, such as PUMP-01. |
| `timestamp_utc` | Measurement time in UTC. |
| `operating_mode` | `STOPPED`, `STARTING`, `RUNNING`, `STOPPING`, or `LOCKED_OUT`. |
| `motor_speed_rpm` | Motor speed in revolutions per minute. Missing on fixed-speed pumps without a drive speed measurement. |
| `motor_current_a` | Motor electrical current in amperes, measured as RMS. |
| `motor_power_kw` | Electrical input power in kilowatts. |
| `suction_pressure_kpa` | Pressure at the pump inlet, in gauge kilopascals. |
| `discharge_pressure_kpa` | Pressure at the pump outlet, in gauge kilopascals. |
| `flow_m3_h` | Water flow in cubic metres per hour. |
| `fluid_temp_c` | Temperature of the pumped liquid, in degrees Celsius. |
| `bearing_temp_c` | Bearing housing temperature, in degrees Celsius. |
| `motor_winding_temp_c` | Temperature inside the motor windings, in degrees Celsius. |
| `vibration_rms_mm_s` | Overall vibration velocity, in millimetres per second, for the 10-1000 Hz band. |
| `vibration_accel_rms_g` | Vibration acceleration RMS, in multiples of standard gravity. |
| `vibration_crest_factor` | Peak acceleration divided by RMS acceleration. Large peaks can indicate impacts. |
| `ambient_temp_c` | Air temperature around the pump, in degrees Celsius. |
| `valve_position_pct` | Valve position as a percentage. |
| `quality_flags` | Combined status flags for the measurement row. Values are explained below. |
| `sensor_source` | Available data sources, separated by `\|`, such as PLC, VFD, WIRELESS, IOLINK, POWER_METER, or FLOW_METER. |
| `sample_interval_s` | Nominal row interval: one second in full data or ten seconds in starter data. |

`quality_flags` is a bitmask: several flags can apply at once. Value 1 means communication loss, 2 means out of range, and 16 means startup inhibit.
Values 4 and 8 mean stale data and bad sensor status. They are reserved and are not set in this release.
A zero flag does not prove that the machine is healthy.

The pump ID and day are stored in folder names in the full telemetry dataset.
Reading the whole telemetry folder with a Parquet dataset reader restores those partition fields. A single file may not contain them.
The `day` partition is the calendar date used to group files.

The `assets.csv` table describes each pump. It also defines the split.
Here, **head** means pressure expressed as the height of a water column. **BEP** means the best-efficiency operating point.

| Field | Meaning |
|---|---|
| `asset_id` | Pump identifier. |
| `split` | `train`, `validation`, or `test`. |
| `site` | Fictional vessel or plant. |
| `system` | The pump's job within that site. |
| `profile` | `SHIP_SW_COOLING`: ship seawater. `SHIP_LT_FRESHWATER`: ship fresh water. `INDUSTRIAL_CW`: industrial cooling water. `PROCESS_TRANSFER`: process-water transfer. |
| `control` | `VFD_TEMPERATURE` or `VFD_PRESSURE`: drive speed follows temperature or pressure. `FLOW_VALVE`: valve-based flow control. `FIXED_SPEED`: fixed motor speed. |
| `instrumentation_tier` | Sensor level: T0, T1, or T2, as explained above. |
| `window_start_utc` | First time in the pump's data window. |
| `window_end_utc` | End of the window. This time is excluded. |
| `commissioned_year` | Installation year. |
| `pump_rated_flow_m3_h` | Flow at BEP and rated speed, in cubic metres per hour. |
| `pump_rated_head_m` | Pump head at BEP and rated speed, in metres. |
| `pump_shutoff_head_m` | Pump head at zero flow and rated speed, in metres. |
| `pump_bep_efficiency` | Pump efficiency at BEP, expressed as a fraction. |
| `npsh3_at_bep_m` | Required inlet pressure margin at BEP, expressed in metres. At this margin, cavitation causes a 3% head reduction. |
| `design_flow_m3_h` | Intended system flow, in cubic metres per hour. |
| `design_head_m` | Intended system head, in metres. |
| `static_head_m` | The fixed head component in the system model, in metres. Zero for closed loops. |
| `motor_rated_kw` | Rated motor output power, in kilowatts. |
| `motor_rated_current_a` | Rated motor current, in amperes. |
| `motor_rated_rpm` | Rated motor speed, in revolutions per minute. |
| `supply_voltage_v` | Electrical supply voltage, in volts. |
| `supply_frequency_hz` | Electrical supply frequency, in hertz. |
| `min_speed_ratio` | Minimum permitted drive speed divided by rated speed. |
| `fluid` | Description of the liquid. |
| `bearing_designation` | Bearing model identifier. |
| `bearing_bpfo_order` | Outer-race defect frequency divided by shaft rotation frequency. |
| `bearing_bpfi_order` | Inner-race defect frequency divided by shaft rotation frequency. |
| `impeller_vanes` | Number of impeller vanes. |
| `vibration_alarm_mm_s` | Configured vibration warning limit, in millimetres per second. |
| `vibration_trip_mm_s` | Configured vibration trip or high-high limit, in millimetres per second. |
| `channels_available` | Names of installed measurement channels, separated by `\|`. |
| `wireless_update_s` | Interval between wireless sensor updates, in seconds. |

The event files contain the official fault labels. Match events to measurements by `asset_id` and time.
Keep physical onset, first detectability, and the scoring window distinct. The scorer uses the rules in [BENCHMARK_RULES.md](../../../BENCHMARK_RULES.md).

| Field | Meaning |
|---|---|
| `event_id` | Identifier for one fault episode. |
| `asset_id` | Pump affected by the event. |
| `event_class` | `MACHINE` for a machine fault or `SENSOR` for an instrument or data problem. |
| `fault_type` | Specific fault name from the list below. |
| `onset_utc` | Physical start of the fault. Earlier starts are clipped to the data window start. |
| `pre_existing` | True if the problem existed before the data window. |
| `detectable_utc` | First time the available sensors show a defined change above noise. Blank means the instruments cannot reveal it. |
| `action_deadline_utc` | First crossing of a conventional alarm or operating limit during the fault. Blank means no crossing within the window. |
| `end_utc` | Repair or resolution time. Blank means the problem continues beyond the window. |
| `severity` | Peak severity: `minor`, `moderate`, or `severe`. |
| `evidence_channels` | Sensor channels that first showed the problem, in order. |
| `observable` | Whether the installed sensors can reveal the fault. Unobservable events are excluded from scoring. |
| `root_cause_event_id` | Link to a causing event. It can also identify a process event or use `OPERATING_CONDITION`. |
| `sensor_tag` | Affected sensor field. `GATEWAY` identifies communication loss. |
| `label_confidence` | Confidence in the label: `HIGH` or `MEDIUM`. Medium includes incomplete or emerging episodes. |

The action deadline is not necessarily the time of mechanical failure. The dataset does not provide a standard remaining-life target for every row.

| Class | Fault name | Meaning |
|---|---|---|
| Machine | `BEARING_WEAR` | Bearing damage develops. |
| Machine | `SUCTION_STRAINER_CLOGGING` | The inlet filter becomes blocked. |
| Machine | `CAVITATION` | Low inlet pressure causes vapour bubbles that collapse. |
| Machine | `DISCHARGE_RESTRICTION` | A valve or heat exchanger restricts outlet flow. |
| Machine | `MISALIGNMENT` | Connected shafts are not correctly aligned. |
| Machine | `IMBALANCE` | Uneven rotating mass causes vibration. |
| Machine | `IMPELLER_WEAR` | Wear reduces the impeller's performance. |
| Machine | `AIR_ENTRAINMENT` | Air enters the pumped liquid. |
| Sensor | `SENSOR_DRIFT` | The sensor reading slowly moves away from the correct value. |
| Sensor | `SENSOR_STUCK` | The sensor keeps reporting a fixed value. |
| Sensor | `SENSOR_CALIBRATION_SHIFT` | The sensor reading acquires an offset. |
| Sensor | `SENSOR_NOISE_SPIKES` | The signal contains excess noise or sudden peaks. |
| Sensor | `COMMUNICATION_LOSS` | Sensor data stop arriving. |

Predictions can also use `UNKNOWN` when the cause is unclear. This can earn detection credit, but no credit for the specific fault type.

The `operating_context.csv` table describes conditions around the pump, once per minute.
These conditions help explain normal changes in measurements.

| Field | Meaning |
|---|---|
| `asset_id` | Pump identifier. |
| `timestamp_utc` | Time of the context record. |
| `schedule_state` | Activity, such as sea passage, port, production, or a process batch. |
| `duty_demand_pct` | Requested cooling or process duty, as a percentage. Values can exceed 100%. |
| `supply_water_temp_c` | Temperature of incoming water, in degrees Celsius. |
| `sea_temp_c` | Seawater temperature, in degrees Celsius. Used for vessels. |
| `ambient_temp_c` | Air temperature in the machinery space, in degrees Celsius. |
| `sea_state_hs_m` | Significant wave height, in metres. Used for vessels. |
| `suction_static_head_m` | Inlet liquid level or submergence, expressed in metres. |
| `controlled_setpoint` | Text describing the control target. |

The `alarms.csv` table records warnings from the existing alarm system. These are observations, not the official fault labels.

| Field | Meaning |
|---|---|
| `asset_id` | Pump identifier. |
| `tag` | Sensor field that triggered the alarm, or `INSTRUMENT` or `GATEWAY`. |
| `alarm_type` | `HI`: high. `HIHI`: high-high. `LO`: low. `BAD_QUALITY`: instrument issue. `COMM_LOSS`: communication loss. |
| `priority` | Priority from 1 to 3. One is highest. |
| `raised_utc` | Time the alarm was raised. |
| `cleared_utc` | Time the alarm cleared, if known. |
| `acknowledged_utc` | Time an operator acknowledged it, if recorded. |
| `value_at_raise` | Sensor value when the alarm was raised. |
| `limit` | Alarm threshold, using the sensor's units. |

The `maintenance.csv` table contains work records. Operator notes can contain incorrect suspicions.
For test pumps, the public file contains only routine history before the measurement window.

| Field | Meaning |
|---|---|
| `record_id` | Work-record identifier. |
| `asset_id` | Pump identifier. |
| `work_type` | Type of inspection or maintenance. |
| `start_utc` | Work start time. |
| `end_utc` | Work end time. |
| `operator_observation` | Text recorded by the operator. |
| `diagnosis_confidence` | `CONFIRMED`, `PROBABLE`, or `UNCERTAIN`. |
| `planned` | True for planned work. False for corrective work. |
| `in_window` | Whether the work occurred inside the measurement window. |

The `features_1min.parquet` table provides calculated sensor summaries. Each row describes one pump during one minute.
The generator groups the raw measurements by pump and minute. It then calculates averages, standard deviations, minimums, maximums, and counts.
This preparation uses fixed calculations. It does not train a machine-learning model.

For example, a complete minute can contain 60 current readings, one per second.
Those readings become five summary values, including `motor_current_a__mean` and `motor_current_a__max`.
The same row also contains summaries for other available sensors.
This reduces the number of rows and can make experiments faster. It also removes detail within each minute.

In the field patterns below, `<channel>` means a numerical sensor name, such as `motor_current_a`.

| Field or field pattern | Meaning |
|---|---|
| `asset_id` | Pump identifier. |
| `window_start_utc` | Start of the one-minute measurement window. |
| `<channel>__mean` | Average of the available measurements. |
| `<channel>__std` | Standard deviation: how much the measurements vary. |
| `<channel>__min` | Lowest measurement. |
| `<channel>__max` | Highest measurement. |
| `<channel>__count` | Number of valid measurements for this channel. |
| `operating_mode` | Most frequent operating state during the minute. |
| `running_fraction` | Fraction of available rows with state `RUNNING`. |
| `quality_any` | Combined quality flags from the available rows. |
| `rows` | Number of telemetry rows in the minute. Full data normally have 60. |

For example, `motor_current_a__mean` is the average current for that minute.
Sparse sensors can have low counts even when telemetry rows are present. Minutes with no telemetry rows are absent.
A summary starting at 12:00 covers measurements before 12:01. Use that complete summary only at or after 12:01.

The `vibration_snapshots.parquet` table provides short recordings for frequency analysis and other signal methods.

| Field | Meaning |
|---|---|
| `asset_id` | Pump identifier. |
| `snapshot_id` | Recording identifier. |
| `timestamp_utc` | Recording time. |
| `sample_rate_hz` | Samples per second: 10,240. |
| `n_samples` | Number of samples: 10,240 for a one-second recording. |
| `axis` | Measurement direction: radial. |
| `unit` | Acceleration unit: `g`, or standard gravity. |
| `rpm_at_capture` | Shaft speed during the recording, in revolutions per minute. |
| `samples` | List of acceleration measurements in time order. |

The `benign_windows_train_validation.csv` table describes normal events that can resemble faults.
An alert in such a period is false if it matches no scored fault event.

| Field | Meaning |
|---|---|
| `asset_id` | Pump identifier. |
| `benign_type` | Type of normal change, from the list below. |
| `start_utc` | Start of the period. |
| `end_utc` | End of the period. |
| `note` | Short explanation of the change. |

The types are `STARTUP`, `SHUTDOWN`, `LOAD_CHANGE`, `SETPOINT_CHANGE`, `STANDBY_CHANGEOVER`, `PLANNED_STOP`, `MAINTENANCE_RESTART`, `HEAVY_WEATHER`, and `VOLTAGE_SAG`.

Your submission is a CSV of warnings for the three test pumps. Use the [prediction template](../../../submission_template/predictions_template.csv).
Its sample rows illustrate the format only. They are not model results.

| Field | Meaning | Required? |
|---|---|---|
| `asset_id` | Pump receiving the warning. | Yes |
| `alert_time_utc` | Time when your system raises the warning, using only information available by then. | Yes |
| `event_class` | Predicted class: `MACHINE` or `SENSOR`. | Yes |
| `fault_type` | Predicted fault name, or `UNKNOWN`. | Yes |
| `confidence` | Your model's confidence, from 0 to 1. | Yes |
| `evidence` | Short explanation of the measurements behind the warning. | No |
| `recommended_action` | Inspection or maintenance suggestion for a human. | No |

The scorer rewards correct detection, early warnings, correct fault types, and correct machine-versus-sensor classification.
It penalises false warnings and excessive repeats. The score is from 0 to 100, not percentage accuracy.
Keep evidence and recommendations in the prototype even though the CSV rules make those fields optional.

Cost analysis combines detector results with separate assumptions about money and maintenance.
These costs are not sensor fields or fault labels. The project includes a business calculator and a simulated intervention comparison.

| Approach or term | Use in this challenge |
|---|---|
| **Cost-benefit model** | Compare avoided downtime and repairs with warning checks and operating costs. |
| **Payback period** | Installation cost divided by positive annual net benefit: years to recover the investment. |
| **Sensitivity analysis** | Change uncertain costs or fault frequency and check whether the conclusion changes. |
| **Cost-based threshold selection: optional extension** | Compare settings on validation using stated costs for missed faults and false warnings. |
| **What-if simulation** | Estimate energy and intervention costs for a proposed action under an assumed fault. |

The supplied [business calculator](../../../examples/business_case.py) estimates whether monitoring one pump could save more money than it costs.
Its use case is a business-case presentation or comparison of cost scenarios after detector performance has been measured.
The benchmark scorer tests detection performance. The calculator takes recall and false-alert rate as inputs and estimates financial outcomes.
It does not test sensor predictions, train a model, or automatically optimise detector settings.
Results can support a choice between detectors or retrofit options under the same assumptions. Automated optimisation would require additional code.

It uses these relationships:

```text
annual net benefit = avoided downtime and repair costs
                     - false-warning handling costs - annual operating costs
payback years = installation cost / positive annual net benefit
```

Additional inputs describe assumed annual fault frequency, downtime, repair costs, installation costs, and operating costs.
It assumes timely, effective action. Recall alone does not prove avoided downtime.
The measured detector rates are inputs. Script defaults differ from the saved [report](../../../examples/business_case_output.md).
Costs are assumptions, not measured savings.

The [what-if example](../../../examples/what_if_example.py) reports daily energy cost and intervention cost, including downtime, labour, and parts.
Its risk score is a proxy, not a failure probability. See [retrofit feasibility](../../../RETROFIT_FEASIBILITY.md) for hardware costs and assumptions.

The remaining files document the dataset itself.

| File | Purpose |
|---|---|
| `data_dictionary.csv` and `schema.json` | Field definitions, types, units, permitted values, and whether fields can be missing. |
| `manifest.json` | Dataset version, tier, file sizes, file hashes, and telemetry row counts for each pump. |
| `units.md` | Unit definitions. |
| `provenance.md` | Generator version, seed information, and configuration origin. |
| `simulation_assumptions.md` | Assumptions used to generate the measurements. |

The [benchmark rules](../../../BENCHMARK_RULES.md) define evaluation. The [technical model](../../../TECHNICAL_MODEL.md) describes the simulation.
The [method dictionary](basic_methods.md) describes applicable statistical and machine-learning approaches.
