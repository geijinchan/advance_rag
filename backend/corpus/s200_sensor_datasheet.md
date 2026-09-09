# S200 Temperature Sensor — Datasheet

**Document ID:** DOC-S200-DS-001 | **Revision:** 2.3 | **Effective date:** 2023-09-18 | **Classification:** Public

**Manufacturer:** IndustriOS Dynamics Ltd., Vondelingenweg 601, 3196 KK Rotterdam, The Netherlands

## 1. Description

The S200 is a loop-powered industrial temperature sensor with a Pt1000 sensing element (IEC 60751 class AA). It measures from -40 °C to +125 °C with an accuracy of +/-0.3 °C over the full range and converts the reading to a linear 4-20 mA signal on a 2-wire loop. The stainless housing (1.4404) is rated IP68, and the electrical interface is a 5-pin M12 A-coded connector, making the sensor suitable for food-adjacent washdown areas and outdoor pump stations.

## 2. Ordering Information

| Part code | Probe length | Connection | Notes |
|---|---|---|---|
| S200-T-125-M12 | 50 mm | M12 5-pin | Standard version |
| S200-T-125-M12-L100 | 100 mm | M12 5-pin | Deep vessels / thermowells |
| S200-T-125-C2 | 50 mm | 2 m fixed cable | Washdown, angled gland |
| S200-T-125-M12-HT | 50 mm | M12 5-pin | High-temperature variant to +150 °C |

## 3. Technical Specifications

| Parameter | Value |
|---|---|
| Measuring range | -40 °C to +125 °C |
| Accuracy | +/-0.3 °C over the full range |
| Sensing element | Pt1000, IEC 60751 class AA |
| Output | 4-20 mA, 2-wire, linear, scaled -40...+125 °C |
| Loop supply voltage | 10-30 VDC |
| Maximum loop load | 500 ohm at 24 V supply |
| Response time | 0.8 s (t63, in flowing water) |
| Self-heating | < 0.05 °C at 1 mA loop current |
| Long-term drift | < 0.1 °C per year |
| Fault signalling | Output driven to 3.5 mA on internal failure |
| Protection rating | IP68 (2 m water column, 30 days continuous) |
| Electrical connection | M12 5-pin A-coded male connector |
| Probe material | Stainless steel 1.4404 |
| Process connection | G1/2 A thread |
| Ambient / media temperature | -40 °C to +125 °C |
| EMC | EN 61326-1 industrial environments |
| Warranty | 18 months (S-Series standard) |

## 4. Electrical Interface

| M12 pin | Signal |
|---|---|
| 1 | Loop supply + (10-30 VDC) |
| 2 | Not connected |
| 3 | 4-20 mA output (to controller AI channel) |
| 4 | Not connected |
| 5 | Shield / drain |

Loop wiring should use shielded 2x0.5 mm2 cable with a maximum length of 100 m. Controllers must interpret loop currents below 3.6 mA as a sensor fault (the S200 drives 3.5 mA on failure); X-Series controllers raise error code E116 for this condition. Torque the M12 coupling to 0.6 Nm — over-torquing cracks the connector insert and voids the IP68 rating.

## 5. Installation Notes

Install the probe so the tip sees the measured medium directly; avoid dead zones near vessel walls. Route sensor cable at least 300 mm away from motor and VFD cables. For outdoor mounting, fit a drip loop before the gland. The sensor requires no warm-up time but allow 10 minutes of media temperature stabilisation before accepting readings used for calibration.

## 6. Calibration and Maintenance

Verify calibration every 12 months per the Maintenance Schedule (DOC-MNT-SC-003): a 1-point check in an ice-water bath at 0.0 °C must read within +/-0.4 °C. If the offset exceeds tolerance, adjust the offset parameter in Device Console (range -1.5 to +1.5 °C). A 2-point verification against a reference thermometer (0 °C and +80 °C) is required after any electronics repair. Long-term drift below 0.1 °C per year normally keeps the sensor within tolerance for three consecutive annual checks.

## 7. Warranty Notes

The S200 carries the S-Series standard 18-month warranty. Water ingress is covered because the device is IP68 — unless the ingress is traced to a damaged or incorrectly torqued connector. Opening the welded housing, third-party recalibration attempts, and media temperatures above +125 °C are exclusions.
