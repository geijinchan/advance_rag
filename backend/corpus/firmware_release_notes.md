# Firmware Release Notes — X-Series Controllers

**Document ID:** DOC-FW-RN-005 | **Revision:** 5.3 | **Maintained by:** Firmware Engineering, IndustriOS Dynamics Ltd., Rotterdam
**Applies to:** X100, X200, X300 controllers

## 1. Release History

| Version | Release date | Status | Applies to |
|---|---|---|---|
| v3.3.0-beta | 2024-05-01 | Beta (opt-in enrollment only) | X200, X300 |
| v3.2.1 | 2024-03-15 | General availability — recommended | X100, X200, X300 |
| v3.2.0 | 2024-01-20 | General availability | X200, X300 |
| v3.1.0 | 2023-11-02 | General availability | X100, X200, X300 |
| v3.0.2 | 2023-06-12 | Maintenance only | X100, X200, X300 |

## 2. v3.1.0 — released 2023-11-02

- Performance: control task throughput improved by 15%, measured on the 64-task ICL benchmark suite (10 ms tick) on X200 reference hardware; X100 gains approximately 12%.
- Fixed a memory leak in the MQTT client that consumed 4 KB per hour on publishers with QoS 1, observed after 30 or more days of uptime.
- MODBUS-RTU: corrected inter-frame gap handling at 38.4 kbaud, which previously caused occasional CRC errors on long bus chains.
- Device Console 3.0 or later is required after upgrading.
- Known open issues at release: none.

## 3. v3.2.0 — released 2024-01-20

- IO-Link support added for the S350 pressure sensor. Master channels are provided by IOM-AIO4 modules (X300) and the X-EBS-IO4 expansion (X200). The Smart Sensor Profile is implemented, with a minimum cycle time of 2.3 ms and full ISDU parameter access.
- Added sensor health telemetry (drift estimate, cycle counters) to the maintenance screen.
- OPC-UA: subscription interval minimum lowered from 250 ms to 100 ms.
- Known open issue at release: issue IOD-4471 (MODBUS-TCP timeout) — fixed in v3.2.1.
- Not available for X100 (no IO-Link master hardware).

## 4. v3.2.1 — released 2024-03-15

- FIXES the MODBUS-TCP timeout bug (issue IOD-4471). Symptom: idle server connections were silently dropped after 3-4 hours, causing SCADA polling gaps and apparent device offline states. Root cause: incorrect keepalive handling on the server socket. The fix implements the documented 30-second keepalive with correct retransmission.
- All MODBUS-TCP installations should upgrade to v3.2.1; installations on v3.2.0 or earlier must not be considered stable for MODBUS-TCP-critical control.
- Configuration compatibility: no migration needed from v3.2.0; projects restore directly.
- Regression suite: 2,147 automated tests, zero failures.

## 5. v3.3.0-beta — released 2024-05-01

- Adds the predictive maintenance module (opt-in): aggregates vibration data from IOM-VIB4 modules and temperature trends to produce a 30-day failure-risk forecast per asset, reported via MQTT topic industrios/site/serial/predict.
- Beta constraints: not validated for safety functions (SIL2 claim unaffected but the module itself is not a safety function), opt-in enrollment via the beta program only, and telemetry data leaves the site through the OT DMZ broker.
- Expected general availability: Q4 2024.
- Feedback and incident reports: beta-program@industrios-dynamics.com.

## 6. Upgrade Path and Precautions

Supported upgrade path: any version from v3.0.2 upward can upgrade directly to v3.2.1. Downgrades are supported only one minor version back (v3.2.1 to v3.2.0). Before every update, create the mandatory full backup described in the Maintenance Schedule (DOC-MNT-SC-003, section 4). The update window is approximately 8 minutes with two automatic reboots; plan downtime accordingly. On the X300, the dual-image mechanism rolls back automatically if signature validation fails, raising error E120 otherwise.

## 7. Package Checksums

| Package | SHA-256 (first 16 hex characters) |
|---|---|
| x100_fw_3.2.1.pkg | 4a91d7be3c05f812 |
| x200_fw_3.2.1.pkg | 9f3c17a82be0d541 |
| x300_fw_3.2.1.pkg | c28b4e6af17d9302 |

Verify checksums in Device Console (Firmware > Verify package) before flashing.
