# S350 Pressure Sensor — Datasheet

**Document ID:** DOC-S350-DS-001 | **Revision:** 1.9 | **Effective date:** 2024-01-25 | **Classification:** Public

**Manufacturer:** IndustriOS Dynamics Ltd., Vondelingenweg 601, 3196 KK Rotterdam, The Netherlands

## 1. Description

The S350 is a piezoresistive industrial pressure sensor for hydraulic, pneumatic, and process-media monitoring. It measures gauge pressure from 0 to 10 bar with an accuracy of +/-0.5% of full scale (including non-linearity, hysteresis, and repeatability). Its primary interface is IO-Link v1.1.3, complemented by two configurable PNP switching outputs. The stainless steel housing is rated IP68 and terminates in a 4-pin M12 connector with a G1/4 male process port.

## 2. Ordering Information

| Part code | Range | Output | Process port |
|---|---|---|---|
| S350-P-10B-IOL | 0-10 bar | IO-Link + 2 switching outputs | G1/4 male |
| S350-P-10B-A | 0-10 bar | 4-20 mA analog, 2-wire | G1/4 male |
| S350-P-16B-IOL | 0-16 bar | IO-Link + 2 switching outputs | G1/4 male |

## 3. Technical Specifications

| Parameter | Value |
|---|---|
| Measuring range | 0-10 bar (gauge) |
| Overpressure limit | 3x full scale (30 bar) |
| Burst pressure | 5x full scale (50 bar) |
| Accuracy | +/-0.5% FS (non-linearity + hysteresis + repeatability) |
| Thermal drift | < 0.02% FS per °C (compensated 0...+50 °C) |
| Response time | < 10 ms (IO-Link cycle dependent, minimum cycle 2.3 ms) |
| Interface | IO-Link v1.1.3, COM2 (38.4 kbaud) |
| Switching outputs | 2x PNP, 200 mA, configurable high/low switch |
| Supply voltage | 18-30 VDC |
| Current consumption | 35 mA typical (without output loads) |
| Rated load cycles | 100 million pressure cycles |
| Protection rating | IP68 |
| Wetted parts | Stainless steel 1.4404 |
| Process connection | G1/4 male, sealing cone per EN 837 |
| Media temperature | -25 °C to +80 °C |
| Vibration resistance | 20 g, 10-2000 Hz per IEC 60068-2-6 |
| Warranty | 18 months (S-Series standard) |

## 4. IO-Link Communication

| M12 pin | Signal |
|---|---|
| 1 | Supply + (18-30 VDC) |
| 2 | IO-Link C/Q (COM2, 38.4 kbaud) |
| 3 | Supply - (GND) |
| 4 | Switching output 2 (DO2, PNP) |

The device follows the Smart Sensor Profile. Process data is 2 bytes of scaled pressure plus 1 status byte; the minimum cycle time is 2.3 ms. Parameters accessible via ISDU include measuring range scaling, averaging filter (0 / 2 / 8 samples), both switching points with hysteresis, and the fault-mode level of the outputs. X-Series controllers require firmware v3.2.0 (2024-01-20) or later for IO-Link master support; X100 controllers do not have IO-Link master channels and must use the analog variant S350-P-10B-A instead.

## 5. Installation

Fit the sensor with 25 Nm torque using an appropriate flat seal (copper or FKM, depending on media). Never grip the housing with pliers — use a spanner on the hex flats only. Keep the media temperature inside the compensated range for best accuracy; outside 0...+50 °C the drift specification no longer applies. Prime the sensor before commissioning to avoid air pockets at the diaphragm, which read as a false offset.

## 6. Calibration and Maintenance

Verify calibration every 12 months (DOC-MNT-SC-003) with a 2-point check: zero at atmospheric pressure and span at 90% of range against a reference gauge of class 0.05 or better. Correct residual offset via the ISDU zero-adjust parameter. Inspect the process seal annually and after every overpressure event above 15 bar. IO-Link communication errors (controller code E119) usually indicate a damaged cable or a loose M12 coupling rather than sensor failure.

## 7. Warranty Notes

The S350 carries the S-Series standard 18-month warranty. Water ingress is covered (IP68) unless caused by connector damage. Exclusions include pressure events above the burst limit, use of unsuitable seals causing media ingress, and unauthorized opening of the sensor electronics.
