# Troubleshooting Guide — LED Patterns, Error Codes, and Escalation

**Document ID:** DOC-TSH-FC-006 | **Revision:** 2.7 | **Effective date:** 2024-03-20 | **Classification:** Customer

**Issued by:** IndustriOS Dynamics Ltd., Rotterdam

## 1. LED Status Patterns

| Status LED pattern | Meaning | First action |
|---|---|---|
| Off | No supply power | Check supply voltage, fuses, and DIN-rail contact |
| Solid green | Normal operation | None |
| Blinking green, 1 Hz | Standby or configuration mode | Expected during commissioning |
| Solid amber | Boot or firmware update in progress | Wait 8 minutes; never cut power during update |
| Blinking amber, 0.5 Hz | Maintenance interval overdue | Perform due task per DOC-MNT-SC-003 |
| Blinking red, 2 Hz | Overtemperature warning (E103 imminent) | Clean filter, verify 100 mm clearance, check ambient |
| Blinking red, 5 Hz | I/O fault on a module or channel | Read error code, check module seating |
| Solid red | Critical fault, outputs disabled | Read error code, power-cycle once |
| Alternating red/amber | Invalid firmware image (E120) | Recovery reflash via USB-C, then contact L2 |

## 2. Error Codes E101-E120

| Code | Meaning | Corrective action |
|---|---|---|
| E101 | Supply undervoltage below tolerance | Check PSU output, wiring, and load budget |
| E102 | Supply overvoltage above tolerance | Check PSU; verify against manual tolerance |
| E103 | Ambient or internal overtemperature | Clean filter, restore clearance, reduce ambient |
| E104 | Internal temperature sensor failed | Power-cycle once; if repeated, RMA |
| E105 | RAM checksum error at boot | Power-cycle; repeated occurrence means RMA |
| E106 | eMMC wear above 90% of rated life | Plan storage replacement via RMA |
| E107 | RTC battery low | Schedule battery replacement within 30 days |
| E108 | Watchdog reset occurred | Reduce task count or fix project bug (DOC-FW-RN-005 v3.1.0) |
| E109 | Power-cycle count exceeds 50,000 rating | Replace controller (end of design life) |
| E110 | MODBUS-TCP connection timeout | Upgrade to firmware v3.2.1; check firewall F1 idle timeout |
| E111 | MODBUS-RTU CRC error | Check cabling, 120 ohm termination, baud rate |
| E112 | I/O module not detected (X300 slot) | Reseat module; see hot-swap procedure DOC-X300-UM-001 section 4 |
| E113 | I/O module type mismatch | Compare slot configuration in Device Console |
| E114 | Digital input stuck at implausible level | Check field wiring and sensor signal |
| E115 | Relay output contact fault | Inspect contacts per DOC-MNT-SC-003 section 3.5 |
| E116 | Loop current below 3.6 mA (S200 fault) | Check sensor loop; replace S200 if wiring is sound |
| E117 | Sensor reading out of range | Verify process conditions; recalibrate |
| E118 | Calibration data invalid | Recalibrate; if rejected, RMA the sensor |
| E119 | IO-Link communication error (S350) | Check M12 coupling and cable; verify master config |
| E120 | Firmware signature verification failed | Recovery reflash; contact L2, quote E120 |

## 3. First-Line Procedure (L1)

1. Record the LED pattern, active error code, firmware version, and operating hours.
2. Power-cycle the device ONCE and allow 3 minutes for a full boot.
3. If the fault persists, export the diagnostics bundle from Device Console (last 500 event-log entries plus I/O snapshot).
4. Check the obvious causes first: supply voltage, connector seating, filter condition, and field wiring.
5. Match the error code against section 2 and apply the corrective action.

## 4. Escalation Path

| Level | Who | Response target | Handles |
|---|---|---|---|
| L1 | Site technician | 4 business hours | LED and error-code interpretation, wiring, power-cycle, filter |
| L2 | IndustriOS support (Rotterdam) | 24 business hours | Firmware issues, configuration, RMA issuance |
| L3 | Engineering (Rotterdam) | 48-72 hours defect analysis | Defect tickets IOD-nnnn, safety-related faults |

Escalate L1 to L2 immediately, skipping the power-cycle, when: error E120 is active, the status LED is solid red after one power-cycle, or any safety function is implicated. Escalate L2 to L3 when a confirmed defect requires a code or hardware change — support issues an IOD ticket and, if the fault affects multiple sites, publishes a field notice within 5 business days.

## 5. Sensor-Specific Checks

S200 (4-20 mA): measure loop current at the controller input. Approximately 3.5 mA means an S200 internal failure (replace the sensor); 0 mA means a broken wire or missing loop supply; a frozen value means the probe is likely fouled — clean and recalibrate.

S350 (IO-Link): check the master port status in Device Console. An E119 code with other devices healthy on the same master points to the sensor cable; E119 on all devices of one master port indicates the port or the X-EBS-IO4 module.
