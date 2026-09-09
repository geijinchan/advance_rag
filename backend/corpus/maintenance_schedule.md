# Maintenance Schedule — X-Series Controllers and S-Series Sensors

**Document ID:** DOC-MNT-SC-003 | **Revision:** 4.1 | **Effective date:** 2024-02-01 | **Classification:** Customer

**Issued by:** IndustriOS Dynamics Ltd., Rotterdam | **Interval basis:** operating hours or calendar time, whichever occurs first

## 1. Maintenance Philosophy

IndustriOS Dynamics products follow a preventive maintenance model. Intervals are defined by operating hours or calendar time, whichever occurs first. Each task is classified by skill level: L (log-only, operator), T (technician), E (electrician). Maintenance records must be retained for at least 3 years (5 years for customers under audit obligations) and must state date, device serial number, operating hours, firmware version, tasks performed, deviations found, and the technician's name.

## 2. Master Schedule

| Interval | Task | Applies to | Skill | Section |
|---|---|---|---|---|
| Every 2000 operating hours or quarterly | Clean or replace cabinet air filter; inspect fan | X100, X200, X300 cabinets | T | 3.1 |
| Quarterly | Review event log for warnings; verify RTC time within +/-2 s | All controllers | L | 3.2 |
| Quarterly | Visual check: connector seating, gasket condition, LED status | All products | L | 3.3 |
| Semi-annually | Terminal torque re-check on a 10% sample | All controllers | E | 3.4 |
| Annually | Relay contact inspection: measure contact resistance | X100, X200; X300 IOM-DO16R | E | 3.5 |
| Annually (every 12 months) | Sensor calibration: S200 1-point; S350 2-point zero/span | S200, S350 | T | 3.6 |
| Annually | Proof test of SIL2 safety functions, record results | X300 | E | 3.7 |
| Annually | eMMC wear-level check and RTC battery voltage test | All controllers | T | 3.8 |
| Every 5 years | Replace RTC battery | All controllers | T | 3.9 |

## 3. Task Details

**3.1 Filter cleaning (every 2000 operating hours).** Power-down is not required. Remove the cabinet filter, wash it in lukewarm water with mild detergent, dry fully, and reinstall. Replace the filter if torn or deformed. A clogged filter is the most common root cause of overtemperature warning E103 in the field.

**3.2 Event log review.** Export the last 90 days of the event log via Device Console and check for recurring warnings (W-PSU, E101 undervoltage dips, E108 watchdog resets). Recurring E108 indicates task overload; reduce the project task count or contact support.

**3.5 Relay contact inspection (annual).** Measure contact resistance across each relay output in the closed state. New contacts measure below 20 milliohm. Replace the relay board (or the IOM-DO16R module on the X300) when any contact exceeds 100 milliohm. Log the measured values per channel.

**3.6 Sensor calibration (every 12 months).** S200: verify at 0 °C in an ice-water bath; the reading must be within +/-0.4 °C, otherwise correct the offset in Device Console. S350: verify zero at atmospheric pressure and span at 90% of range against a class 0.05 reference gauge; correct via the ISDU zero-adjust parameter. Calibration drift beyond the correction range requires an RMA.

**3.7 SIL2 proof test (X300, annual).** Exercise every safety function end-to-end (trigger the E-stop, verify the output reaches the safe state within the specified time) and archive the result with the safety key signature. A missed proof test suspends the SIL2 claim until completed.

## 4. Firmware Update Procedure

Before every firmware update, create a full backup: project file, parameter set, calibration offsets, and the event log — to the service laptop and to the Customer Portal. Verify the backup by restoring it to the Device Console emulator before flashing. The update takes about 8 minutes; the controller reboots twice. X300 units keep a dual firmware image and roll back automatically if validation fails. Downgrades are supported only to the previous minor version. Always update during planned downtime and never during a running batch or an active safety proof test.

## 5. Records and Audit

The maintenance log is the primary evidence in warranty discussions (DOC-WAR-POL-001) and in functional-safety audits (DOC-SAF-CG-002). Missing relay inspection or sensor calibration records can void claims where the fault mode is plausibly related to the skipped task.
