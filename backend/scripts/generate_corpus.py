#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_corpus.py — Build the demo knowledge-base corpus for the
IndustriOS Dynamics Ltd. "Document Intelligence Assistant" RAG service.

Produces (relative to mini-services/rag-agent/):

  corpus/x100_controller_manual.md          (~700 words, markdown)
  corpus/x200_controller_manual.md          (~750 words, markdown, links chart asset)
  corpus/x300_controller_manual.md          (~700 words, markdown)
  corpus/s200_sensor_datasheet.md           (~650 words, markdown)
  corpus/s350_sensor_datasheet.md           (~650 words, markdown)
  corpus/warranty_policy.md                 (~750 words, markdown)
  corpus/safety_compliance_guide.md         (~700 words, markdown)
  corpus/maintenance_schedule.md            (~700 words, markdown)
  corpus/firmware_release_notes.md          (~800 words, markdown)
  corpus/network_integration_guide.md       (~800 words, markdown)
  corpus/rma_faq.txt                        (~650 words, plain text Q&A)
  corpus/troubleshooting_flowchart.md       (~750 words, markdown)
  corpus/deployment_checklist.md            (~700 words, markdown)
  corpus/x200_manual.pdf                    (typeset from x200_controller_manual.md, embedded chart)
  corpus/warranty_policy.pdf                (typeset from warranty_policy.md)
  corpus/firmware_release_notes.pdf         (typeset from firmware_release_notes.md)
  corpus/assets/warranty_duration.png       (matplotlib bar chart, low-saturation palette)

Usage:
    /home/z/.venv/bin/python3 scripts/generate_corpus.py     (from rag-agent/)

The script is idempotent: every artefact is overwritten on each run.
"""

from __future__ import annotations

import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent      # .../mini-services/rag-agent
CORPUS_DIR = BASE_DIR / "corpus"
ASSETS_DIR = CORPUS_DIR / "assets"

COMPANY = "IndustriOS Dynamics Ltd."
ADDRESS = "Vondelingenweg 601, 3196 KK Rotterdam, The Netherlands"
SUPPORT_MAIL = "support@industrios-dynamics.com"
SUPPORT_PHONE = "+31 10 555 0142"

# ============================================================================
#  DOCUMENT CONTENT (markdown / plain-text sources)
# ============================================================================

X100_MANUAL = """
# X100 Industrial Controller — User Manual

**Document ID:** DOC-X100-UM-001 | **Revision:** 4.2 | **Effective date:** 2024-01-10 | **Classification:** Customer

**Manufacturer:** IndustriOS Dynamics Ltd., Vondelingenweg 601, 3196 KK Rotterdam, The Netherlands
**Support:** support@industrios-dynamics.com | +31 10 555 0142 | Mon-Fri 08:00-17:00 CET

## 1. Product Overview

The X100 is the entry-level controller of the IndustriOS Dynamics X-Series. It is designed for small machine control, building automation cells, and pilot lines where a compact DIN-rail unit is sufficient. The controller executes the IndustriOS Control Logic (ICL) runtime and supports up to 64 concurrent control tasks on a 10 ms scheduler tick. More than 41,000 units have shipped since the platform launched in March 2019, and the hardware is qualified for 50,000 power cycles.

Typical applications include conveyor sequencing, packaging machines with up to 12 end positions, and pump-station control in municipal water treatment. For installations that need more I/O, a wider temperature window, or an extended warranty, the X200 and X300 controllers described in their own manuals are the recommended alternatives.

## 2. Technical Specifications

| Parameter | Value |
|---|---|
| Part number | X100-24D-12R |
| Supply voltage | 24 VDC +/- 10% (18.0-26.4 VDC) |
| Processor | ARM Cortex-A53 quad-core @ 1.2 GHz |
| Memory | 2 GB DDR4 RAM, 16 GB eMMC flash |
| Digital inputs | 12 channels, 24 VDC sink/source, 3.5 ms filter |
| Relay outputs | 8 channels, SPDT, 250 VAC / 30 VDC, 6 A per channel |
| Protection rating | IP54 (front panel), IP20 (terminal compartment) |
| Operating temperature | -10 °C to +55 °C, non-condensing |
| Storage temperature | -25 °C to +70 °C |
| Relative humidity | 5-95% RH, non-condensing |
| Communication | 2x Ethernet 10/100 (MODBUS-TCP, OPC-UA), RS-485 (MODBUS-RTU) |
| Service interface | USB-C (configuration only, not a fieldbus) |
| Dimensions / weight | 110 x 90 x 62 mm, 420 g |
| Power consumption | 11 W typical, 15 W maximum |
| Standard warranty | 12 months from delivery date |

The X100 is not eligible for extended warranty registration. The 12-month term is fixed at the point of sale; customers with longer coverage horizons should select the X200 (24 months, extendable to 36) or the X300 (36 months standard).

## 3. Mechanical Installation

Mount the controller on a 35 mm DIN rail (EN 60715) inside a control cabinet, keeping a minimum clearance of 100 mm on all sides for convection cooling. Vertical side mounting is permitted but reduces the permissible ambient temperature by 5 °C. Do not install the unit directly above heat sources such as braking resistors or soft starters; radiated heat can push the cabinet interior beyond the +55 °C operating limit and trigger overtemperature warning code E103.

The terminal compartment is rated IP20 and must remain inside a closed enclosure. Torque terminal screws to 0.5-0.6 Nm using a calibrated driver, and re-check torque after the first 100 operating hours and annually thereafter, as specified in the Maintenance Schedule (DOC-MNT-SC-003).

## 4. Electrical Installation

All wiring must comply with EN 60204-1. Use ferruled conductors of 0.25-2.5 mm2 cross-section. The digital inputs accept both sink and source logic with a common reference on terminal X1:9. Relay outputs are voltage-free contacts; for inductive loads up to 2 A, fit an RC snubber (47 ohm / 100 nF) across each contact to limit arcing and contact wear. Relay contacts are rated for 100,000 mechanical cycles at rated load.

Provide a 24 VDC supply with at least 30 W headroom for the controller plus its I/O budget. Redundant feeding is not supported on this model; applications that require redundant power must specify the X300, which ORs two independent feeds.

## 5. Commissioning Quick Start

1. Verify supply polarity and magnitude (18.0-26.4 VDC) before energising.
2. Connect a service laptop (link-local address 169.254.0.10/16) to the USB-C service port.
3. Launch IndustriOS Device Console 3.1 or later and accept the new device.
4. Assign a static IP on the control network as described in the Network Integration Guide (DOC-NET-IG-003).
5. Load the ICL project file and set the watchdog mode to "Safe state".
6. Execute the point-to-point I/O test, then release forces and enable outputs.
7. Record firmware version and project checksum in the Deployment Checklist (DOC-DEP-CL-004).

## 6. Warranty and Service Summary

The X100 carries a 12-month standard warranty from the delivery date. Claims are handled through the central RMA process: evaluation within 48 hours of receipt at the Rotterdam facility and repair within 5-7 working days. Water damage is excluded for devices that are not rated IP68 — the X100's IP54 rating means splash-zone or outdoor installations void the warranty. See the Warranty Policy (DOC-WAR-POL-001) for the complete exclusion list.

## 7. Related Documents

- Warranty Policy (DOC-WAR-POL-001)
- Safety and Compliance Guide (DOC-SAF-CG-002)
- Maintenance Schedule (DOC-MNT-SC-003)
- Deployment Checklist (DOC-DEP-CL-004)
- Firmware Release Notes (DOC-FW-RN-005)
"""

X200_MANUAL = """
# X200 Industrial Controller — User Manual

**Document ID:** DOC-X200-UM-001 | **Revision:** 5.1 | **Effective date:** 2024-04-02 | **Classification:** Customer

**Manufacturer:** IndustriOS Dynamics Ltd., Vondelingenweg 601, 3196 KK Rotterdam, The Netherlands
**Support:** support@industrios-dynamics.com | +31 10 555 0142 | Mon-Fri 08:00-17:00 CET

## 1. Product Overview

The X200 is the mid-range X-Series controller for multi-axis material handling, filling and bottling lines, and retrofit programs. It doubles the I/O budget of the X100 with 24 digital inputs and 16 relay outputs, and widens environmental tolerance to IP65 and -20 °C to +60 °C, which permits installation near wash-down zones and in unheated plant rooms.

The X200 standard warranty is 24 months from the delivery date. Registering the device within 30 days of purchase extends the warranty to 36 months at no additional cost. Registration takes place in the IndustriOS Customer Portal: enter the serial number (format IOS-XX-NNNNNN) and the delivery date printed on the invoice; confirmation arrives by e-mail within two working days. Devices registered on day 31 or later remain covered for the standard 24 months only — the 30-day window is enforced without exceptions.

![Standard and extended warranty duration by model](assets/warranty_duration.png)

## 2. Technical Specifications

| Parameter | Value |
|---|---|
| Part number | X200-48D-16R |
| Supply voltage | 48 VDC +/- 10% (43.2-52.8 VDC) |
| Processor | ARM Cortex-A72 quad-core @ 1.5 GHz |
| Memory | 4 GB DDR4 RAM, 32 GB eMMC flash |
| Digital inputs | 24 channels, 24 VDC sink/source, configurable 0.5/3.5/10 ms filter |
| Relay outputs | 16 channels, SPDT, 250 VAC / 30 VDC, 6 A per channel |
| Protection rating | IP65 (front face and sealed housing) |
| Operating temperature | -20 °C to +60 °C, non-condensing |
| Storage temperature | -30 °C to +75 °C |
| Relative humidity | 5-95% RH, non-condensing |
| Communication | 2x Gigabit Ethernet (MODBUS-TCP, OPC-UA), RS-485 (MODBUS-RTU), CAN 2.0B |
| Expansion | 2x X-EBS bus ports, up to 64 additional I/O channels |
| Dimensions / weight | 160 x 120 x 85 mm, 980 g |
| Power consumption | 18 W typical, 24 W maximum |
| Standard warranty | 24 months, extendable to 36 months with 30-day registration |

## 3. Firmware Requirements

Install firmware v3.2.1 (released 2024-03-15) or later before commissioning. This release fixes the MODBUS-TCP timeout bug tracked as issue IOD-4471, in which idle server connections were silently dropped after 3-4 hours and produced SCADA polling gaps. Firmware v3.2.0 and earlier must not be used in MODBUS-TCP-critical installations. Always create a full project and parameter backup before updating firmware (see DOC-MNT-SC-003); the update itself takes about 8 minutes and reboots the controller twice.

## 4. Mechanical and Electrical Installation

Mount on a 35 mm DIN rail (EN 60715) with at least 100 mm clearance on all sides. The IP65 rating depends on the front gasket being correctly seated; inspect the gasket whenever the cover is removed and replace it after five opening cycles. Terminal torque is 0.6-0.8 Nm with 0.25-2.5 mm2 ferrules.

The 48 VDC supply must be a SELV/PELV circuit per EN 61010-1. Inrush current at power-up is 1.9 A for 50 ms; protect the feed with a 2 A slow-blow fuse. Relay contacts are rated for 100,000 mechanical cycles at rated load; apply RC snubbers (47 ohm / 100 nF) on inductive loads as on the X100. Separate signal wiring from motor and VFD cables by at least 300 mm to avoid EMC-induced input glitches.

## 5. Commissioning

1. Confirm supply voltage 43.2-52.8 VDC and protective-earth continuity below 0.1 ohm.
2. Connect via the USB-C service port (laptop link-local address 169.254.0.10/16) and Device Console 3.1 or later.
3. Set the static control-network IP and VLAN tag (default VLAN 10) per DOC-NET-IG-003.
4. Flash firmware v3.2.1 or later and restore the project backup.
5. Run the 24-input point-to-point test and the 16-output forced test.
6. Verify MODBUS-TCP connectivity on port 502 from the SCADA host and OPC-UA on port 4840.
7. Register the device in the Customer Portal for the warranty extension and file the Deployment Checklist (DOC-DEP-CL-004).

## 6. Warranty Summary

| Coverage | Duration | Condition |
|---|---|---|
| Standard warranty | 24 months from delivery | Automatic |
| Extended warranty | 36 months from delivery | Registration within 30 days of purchase |
| Out-of-warranty repair | Quoted per repair | Evaluation free of charge |

Full terms, exclusions, and the RMA process appear in the Warranty Policy (DOC-WAR-POL-001). Water damage on non-IP68 equipment is excluded; although the X200 is IP65, immersion and high-pressure jets aimed at the gasket are not covered.

## 7. Related Documents

- Warranty Policy (DOC-WAR-POL-001)
- Network Integration Guide (DOC-NET-IG-003)
- Troubleshooting Guide (DOC-TSH-FC-006)
- Firmware Release Notes (DOC-FW-RN-005)
"""

X300_MANUAL = """
# X300 Flagship Controller — User Manual

**Document ID:** DOC-X300-UM-001 | **Revision:** 3.0 | **Effective date:** 2024-02-20 | **Classification:** Customer

**Manufacturer:** IndustriOS Dynamics Ltd., Vondelingenweg 601, 3196 KK Rotterdam, The Netherlands
**Support:** support@industrios-dynamics.com | +31 10 555 0142 | Mon-Fri 08:00-17:00 CET

## 1. Product Overview

The X300 is the flagship of the X-Series, intended for safety-rated and high-availability process control: steel handling, tunnel ventilation, pharmaceutical batch control, and offshore utility skids. It is certified for SIL2 (IEC 61508) safety functions and carries a 36-month warranty as standard. The platform introduces redundant power supplies and hot-swappable I/O modules, allowing channel-level maintenance without process interruption.

Where the X100 and X200 use fixed onboard I/O, the X300 hosts up to 8 IOM-X modules in local slots (each module provides 16 channels: IOM-DI16, IOM-DO16R, IOM-AIO4, or IOM-VIB4). Two X-EBS-2 fibre expansion racks raise the maximum to 384 channels. Module replacement under power is qualified for 5,000 insertion cycles per slot.

## 2. Technical Specifications

| Parameter | Value |
|---|---|
| Part number | X300-48D-SIL2 |
| Supply voltage | 2x independent feeds, 24 or 48 VDC +/- 10%, diode-ORed |
| Processor | ARM Cortex-A78 quad-core @ 2.0 GHz |
| Memory | 8 GB DDR4 ECC RAM, 64 GB industrial pSLC eMMC |
| Safety certification | SIL2 per IEC 61508; PL d, Category 3 per ISO 13849-1 |
| Protection rating | IP67 (fully sealed housing) |
| Operating temperature | -25 °C to +65 °C, non-condensing |
| Storage temperature | -40 °C to +80 °C |
| Local I/O | 8 hot-swappable IOM-X module slots, 16 channels each |
| Expansion | 2x X-EBS-2 fibre racks, 384 channels total |
| Communication | 2x Gigabit Ethernet, 2x 10G SFP+ (option), RS-485, CAN FD |
| Redundancy | Dual PSU, RAID-1 flash mirror, dual watchdog |
| Dimensions / weight | 310 x 210 x 130 mm, 3.1 kg |
| Power consumption | 35 W typical, 48 W with full I/O load |
| Standard warranty | 36 months from delivery (no registration required) |

## 3. Redundant Power Supplies

The X300 accepts two independent supply feeds, PSU-A and PSU-B, each 24 or 48 VDC within 10% tolerance. The feeds are diode-ORed internally; automatic switchover on loss of one feed takes less than 1 ms and does not interrupt task execution. Each feed must be protected with a 4 A slow-blow fuse and wired from a separate circuit, ideally from separate distribution boards, so that a single board outage cannot take down the controller.

Loss of either feed raises warning W-PSU in the event log (this is not a safety fault); the controller continues indefinitely on the remaining feed. PSU voltages are visible on the Device Console diagnostics page and via OPC-UA (nodes ns=4;s=Power.PSU_A_V and ns=4;s=Power.PSU_B_V).

## 4. Hot-Swappable I/O Modules

I/O modules can be replaced with the system powered and running. Replacement procedure:

1. Confirm the module reports no active fault (solid green module LED, no E112/E113 for that slot).
2. Open the front latch of the module carrier.
3. Pull the module straight out within 5 seconds.
4. Insert the replacement module within 60 seconds, before the channel timeout expires.
5. The controller auto-detects the module and copies the stored slot configuration; no project reload is needed.
6. Confirm the module LED returns to solid green within 10 seconds.

Safety-relevant channels transition to their defined safe state during the swap, for a maximum of 500 ms. Never swap modules while a proof test of safety functions is in progress.

## 5. Safety Functions (SIL2)

The X300 is certified for SIL2 (IEC 61508) safety functions. Typical implemented functions include emergency stop, overspeed protection, and overtemperature interlock. The safety task executes on a dedicated core with a 5 ms cycle. The certification corresponds to Performance Level d, Category 3 (dual-channel, monitored) per ISO 13849-1. The proof-test interval is 12 months — see the Maintenance Schedule (DOC-MNT-SC-003), task 3.7.

Safety parameters are stored in a write-protected area; modification requires the physical safety key delivered with the device. Unauthorized modification voids both the warranty and the SIL certificate. The X100 and X200 are not safety-rated and must never be used as the sole safety channel.

## 6. Warranty and Service

The X300 carries a 36-month standard warranty from the delivery date; no registration is required. RMA handling is identical to other products — 48-hour evaluation and 5-7 working day repair at the Rotterdam facility — except that safety-certified repairs may take up to 10 working days because the SIL2 certificate chain requires a documented final test and sign-off by the compliance engineer. Unauthorized opening of the sealed housing is an exclusion that additionally invalidates the SIL certificate.

## 7. Related Documents

- Safety and Compliance Guide (DOC-SAF-CG-002)
- Maintenance Schedule (DOC-MNT-SC-003)
- Network Integration Guide (DOC-NET-IG-003)
- Warranty Policy (DOC-WAR-POL-001)
"""

S200_DATASHEET = """
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
"""

S350_DATASHEET = """
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
"""

WARRANTY_POLICY = """
# Warranty Policy — X-Series Controllers and S-Series Sensors

**Document ID:** DOC-WAR-POL-001 | **Revision:** 6.0 | **Effective date:** 2024-01-01 | **Classification:** Customer

**Issued by:** IndustriOS Dynamics Ltd., Vondelingenweg 601, 3196 KK Rotterdam, The Netherlands
**Applies to:** all products delivered from 2019-06-01 onwards

## 1. Purpose and Scope

This policy defines the warranty terms for IndustriOS Dynamics Ltd. hardware: X-Series controllers (X100, X200, X300), S-Series sensors (S200, S350), and original accessories (IOM-X modules, X-EBS bus components, connectors). The warranty period starts on the delivery date (Incoterms DAP) recorded on the invoice. The warranty is transferable: if the device is resold, the remaining term continues from the original delivery date, and the new owner may register ownership in the Customer Portal.

## 2. Warranty Duration by Product

| Product | Standard warranty | Extended warranty | Registration window |
|---|---|---|---|
| X100 controller | 12 months | not available | - |
| X200 controller | 24 months | 36 months | 30 days from purchase |
| X300 controller | 36 months | not offered (already maximum) | - |
| S200 sensor | 18 months | not available | - |
| S350 sensor | 18 months | not available | - |

## 3. Extended Warranty Registration (X200 only)

The X200 extended warranty extends coverage from 24 to 36 months at no charge, provided the device is registered within 30 days of purchase. Registration is completed in the IndustriOS Customer Portal by entering the serial number (format IOS-XX-NNNNNN) and the delivery date from the invoice; the portal sends an activation confirmation by e-mail within two working days. The 30-day window is calculated from the purchase date, not the installation or commissioning date. Registrations submitted on day 31 or later are rejected automatically — there is no grace period and no manual override. The extended term applies only to units purchased new from IndustriOS Dynamics or an authorised distributor.

## 4. Exclusions

Warranty claims are denied for the following causes:

- Water damage to devices not rated IP68. The X100 (IP54) and X200 (IP65) are excluded for immersion, jet, and splash damage; S-Series sensors (IP68) are covered unless ingress is traced to a damaged or incorrectly torqued connector.
- Unauthorized repair, modification, or opening of sealed housings by anyone other than IndustriOS Dynamics or an authorised service partner. For the X300 this additionally voids the SIL2 certificate.
- Supply voltage, overvoltage, or lightning events beyond the tolerance stated in the product manual (for example, outside 43.2-52.8 VDC on the X200).
- Use of unsupported firmware: pre-release or beta firmware outside the enrolled beta program.
- Wear beyond rated lifetime: relay contacts (100,000 cycles), RTC batteries (5 years), and gaskets (five opening cycles).
- Installation contrary to the manuals: insufficient clearance, missing filters, ambient temperatures above the specified limits.
- Cosmetic damage and faults caused by customer-loaded project software.
- Shipping damage where the carrier was chosen by the customer against packaging instructions (see DOC-RMA-FAQ-001).

## 5. RMA Process

1. Obtain an RMA number via the Customer Portal or support@industrios-dynamics.com, providing the serial number, purchase order, firmware version, and a fault description. RMA numbers are issued within 4 business hours and are valid for 30 days.
2. Ship the unit to the Rotterdam facility (Vondelingenweg 601, 3196 KK Rotterdam, The Netherlands). In-warranty EU customers receive a prepaid return label.
3. Evaluation is completed within 48 hours of receipt at goods-in. The customer receives a report stating the confirmed fault, the warranty decision, and — for out-of-warranty units — a repair quote.
4. In-warranty repairs are completed within 5-7 working days after evaluation. Spare parts for all models shipped since 2019 are stocked in Rotterdam.
5. The repaired unit is returned with a 90-day warranty on the repair itself.

Dead-on-arrival units (failure within 14 days of delivery) qualify for an advance replacement: a refurbished or new unit ships before the defective unit is returned.

## 6. Out-of-Warranty Service

Evaluation of out-of-warranty units is free of charge. Repairs are quoted before any work starts (185 EUR per hour labour plus parts) and carry the same 90-day repair warranty. Loaner units from the X200/X300 pool are available for a maximum of 21 days, subject to availability; the 150 EUR administration fee is waived for customers with a confirmed repair order.

## 7. Claims Documentation

A complete claim contains: proof of purchase, the serial number, a fault description with the first occurrence date, the firmware version, the last maintenance record, and photographs for shipping-damage claims. IndustriOS Dynamics responds to warranty claims within 5 business days. Fraudulent claims (for example, serial numbers transplanted between housings) void the warranty of all devices involved and may be reported to the authorities.
"""

SAFETY_GUIDE = """
# Safety and Compliance Guide — X-Series and S-Series Products

**Document ID:** DOC-SAF-CG-002 | **Revision:** 3.5 | **Effective date:** 2023-12-05 | **Classification:** Customer

**Issued by:** IndustriOS Dynamics Ltd., Rotterdam | **Safety hotline:** +31 10 555 0199 (24/7)

## 1. CE and Regulatory Markings

All X-Series controllers and S-Series sensors carry the CE mark. The applicable EU directives are: Low Voltage Directive 2014/35/EU, EMC Directive 2014/30/EU, Machinery Directive 2006/42/EC (controllers delivered as safety components), and RoHS Directive 2011/65/EU. Declarations of Conformity per model are downloadable from the Customer Portal. For the UK market, the equivalent UKCA marking and declaration apply. EMC compliance assumes installation per EN 60204-1 with signal/motor cable separation of at least 300 mm.

## 2. Functional Safety: SIL Levels and Performance Levels

Functional safety capability is defined per IEC 61508 (Safety Integrity Levels, SIL) and ISO 13849-1 (Performance Levels, PL):

| SIL (IEC 61508) | Probability of dangerous failure per hour | IndustriOS product coverage |
|---|---|---|
| SIL1 | 10^-6 to 10^-5 /h | Not offered as a standalone claim |
| SIL2 | 10^-7 to 10^-6 /h | X300 safety functions (certified) |
| SIL3 | 10^-8 to 10^-7 /h | Not claimed by any IndustriOS product |

The X300 implements safety functions at SIL2, which corresponds to Performance Level d, Category 3 (dual-channel, monitored) per ISO 13849-1. The safety task runs on a dedicated core with a 5 ms cycle, and the proof-test interval is 12 months. The X100 and X200 are general-purpose controllers with no safety certification: they must never be used as the sole channel of a safety function. If an application requires PL d with non-safety controllers in the signal path, an external safety relay must interrupt the energy path.

## 3. Lockout-Tagout Procedure (LOTO)

Apply the following six-step procedure before any mechanical or electrical intervention:

1. Notify all affected personnel and record the planned intervention in the shift log.
2. Identify every energy source: mains supply, 24/48 VDC auxiliary feeds, compressed air (6 bar nominal), hydraulics, counterweight or spring energy, and UPS buffer batteries.
3. Isolate each source at its dedicated isolation point: main switch Q1, pneumatic manifold valve, hydraulic block valve, and the UPS maintenance bypass.
4. Apply one personal lock and tag per technician to each isolation point. For teams, all locks go on a group lockbox hasp; the last lock removed releases the isolation.
5. Verify the zero-energy state: confirm absence of voltage with a calibrated meter (CAT III 600 V minimum), bleed residual air to 0 bar, and wait at least 5 minutes for DC-bus capacitors to discharge below 60 VDC.
6. Release in the documented order: each lock is removed only by its owner, re-energisation follows the start-up sequence, and the hand-over is countersigned in the shift log.

## 4. Personal Protective Equipment

| Task | Required PPE | Standard |
|---|---|---|
| General panel and field work | Safety glasses with side shields | EN 166 |
| Work on circuits above 50 VAC or 120 VDC | Class 0 insulating gloves (1000 V), retested every 6 months | EN 60903 |
| Work near live switchgear | Arc-flash clothing, minimum 8 cal/cm2 | NFPA 70E, category 2 |
| Areas above 85 dB(A) | Hearing protection | EN 352 |
| Rotating equipment nearby | Close-fitting clothing, tied-back hair, no gloves near rotating parts | - |

## 5. Electrical Safe Working Rules

Live working is prohibited unless a documented risk assessment approves it and a competent supervisor is present (two-person rule). Test instruments must be CAT III 600 V rated as a minimum for controller cabinets. Before energising a modified cabinet, verify protective-earth continuity below 0.1 ohm and insulation resistance above 1 megohm between power conductors and the enclosure. Never defeat interlocks or door switches — defeat of a safety interlock must be treated as a reportable incident.

## 6. Incident Reporting

Report all safety incidents, near-misses, and defeated interlocks within 24 hours to safety@industrios-dynamics.com or the 24/7 hotline +31 10 555 0199. Serious injuries involving hospitalisation must be reported immediately by phone. Incident reports are mandatory inputs to the quarterly safety review and to any SIL2 proof-test audit trail.
"""

MAINTENANCE_SCHEDULE = """
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
"""

FIRMWARE_NOTES = """
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
"""

NETWORK_GUIDE = """
# Network Integration Guide — X-Series Controllers

**Document ID:** DOC-NET-IG-003 | **Revision:** 3.4 | **Effective date:** 2024-03-01 | **Classification:** Customer

**Issued by:** IndustriOS Dynamics Ltd., Rotterdam

## 1. Architecture Overview

IndustriOS recommends a three-zone OT network. VLAN 10 (control) contains controllers, remote I/O, and SCADA servers. VLAN 20 (OT DMZ) contains the MQTT broker, historian, and the engineering jump host. VLAN 30 (enterprise) contains workstations and business systems. Controllers live only in VLAN 10; all north-bound traffic passes through the OT DMZ. Direct enterprise-to-controller routing is prohibited — enterprise users reach devices only through the jump host in VLAN 20.

## 2. Addressing and Static IP Setup

The factory default address is 192.168.10.10/24 with gateway 192.168.10.1. DHCP client mode is supported but static addressing is required for control devices (recommendation NET-01), because SCADA configurations and firewall rules pin controller addresses. Set the static address in Device Console (Network > Interfaces) or via the USB-C service port, where the controller always answers on link-local 169.254.0.10/16 regardless of the configured production address. Address changes take effect after a controller reboot. Document the assigned address, VLAN, and MAC address (format 00:1B:2F:nn:nn:nn) in the Deployment Checklist.

## 3. MODBUS-TCP

- TCP port 502 (fixed; cannot be changed).
- Concurrent client connections: 8 on the X100, 16 on the X200, 32 on the X300.
- Unit ID range 1-247, default 1, configurable per device.
- Default response timeout 3000 ms; keepalive 30 s.
- Maximum 125 registers per read request.

Firmware v3.2.1 (2024-03-15) or later is required for stable long-lived connections: v3.2.0 and earlier contain issue IOD-4471, which drops idle connections after 3-4 hours. Where a NAT or stateful firewall sits between client and controller, set the firewall idle timeout above the 30-second keepalive interval.

## 4. OPC-UA

- Endpoint: opc.tcp://<controller-address>:4840.
- Security policy Basic256Sha256 with message signing; the security policy "None" is rejected by default and must stay disabled in production.
- Anonymous access is disabled by default. Built-in users: opc_reader (read-only) and opc_admin (read/write); passwords are set at commissioning.
- Session timeout 60 s; minimum subscription interval 100 ms (X200/X300; X100 minimum 250 ms).

## 5. MQTT

Controllers act as MQTT clients and publish to a broker in the OT DMZ (VLAN 20). Production connections must use TLS on port 8883 with server-certificate verification and client certificates; plaintext port 1883 is acceptable only in closed laboratory networks.

- Topic scheme: industrios/<site>/<serial>/telemetry (QoS 1, retain=false) and industrios/<site>/<serial>/status (retain=true).
- Keepalive 60 s. Last Will and Testament publishes "offline" to the status topic.
- Example broker configuration (Mosquitto 2.x): listener 8883, protocol mqtt, tls_version tlsv1.2, require_certificate true, allow_anonymous false.
- Publish payload: JSON, one sample frame per message, device clock timestamp in ISO 8601.

## 6. VLAN Segmentation

| VLAN | ID | Example subnet | Contains |
|---|---|---|---|
| Control | 10 | 10.10.10.0/24 | Controllers, remote I/O, SCADA servers |
| OT DMZ | 20 | 10.10.20.0/24 | MQTT broker, historian, jump host |
| Enterprise | 30 | 10.10.30.0/24 | Workstations, ERP, user laptops |

Trunks between switches use 802.1Q tagging; access ports in VLAN 10 are configured with port security (maximum 4 MAC addresses). Enable IGMP snooping on control switches to contain multicast discovery traffic. Routing between VLAN 10 and VLAN 20 is restricted to the listed firewall rules below; VLAN 30 to VLAN 10 is denied entirely.

## 7. Firewall Rules

| Rule | Source | Destination | Port | Protocol | Action | Purpose |
|---|---|---|---|---|---|---|
| F1 | SCADA servers (VLAN 10) | Controllers (VLAN 10) | 502 | TCP | allow | MODBUS-TCP |
| F2 | Controllers (VLAN 10) | OPC-UA server (VLAN 10) | 4840 | TCP | allow | OPC-UA |
| F3 | Controllers (VLAN 10) | MQTT broker (VLAN 20) | 8883 | TCP | allow | MQTT over TLS |
| F4 | Jump host (VLAN 20) | Controllers (VLAN 10) | 22 | TCP | allow | SSH service access |
| F5 | Controllers (VLAN 10) | NTP server (VLAN 20) | 123 | UDP | allow | Time synchronisation |
| F6 | Monitoring (VLAN 20) | Controllers (VLAN 10) | 161 | UDP | allow | SNMP read-only |
| F7 | Any | Controllers (VLAN 10) | any | any | deny | Default deny |

## 8. Hardening Checklist

- Disable unused protocols (for example CAN or RS-485) on each device.
- Replace the self-signed OPC-UA certificate with the site CA certificate at commissioning.
- Run firmware v3.2.1 or later (MODBUS-TCP fix IOD-4471).
- Enable SNMP read-only community strings of at least 16 random characters.
- Review switch port-security and VLAN assignment quarterly per DOC-MNT-SC-003.
"""

RMA_FAQ = r"""
================================================================================
  INDUSTRIOS DYNAMICS LTD.
  RMA (RETURN MERCHANDISE AUTHORIZATION) - FREQUENTLY ASKED QUESTIONS
  Document ID: DOC-RMA-FAQ-001 | Revision 1.8 | Effective date: 2024-02-01
  Applies to: X-Series controllers (X100/X200/X300), S-Series sensors (S200/S350)
================================================================================


Q1. HOW DO I REQUEST AN RMA NUMBER?
A1. Submit a request through the Customer Portal (Support > RMA > New Request)
    or e-mail support@industrios-dynamics.com with the following information:
    device serial number (format IOS-XX-NNNNNN), purchase order or invoice
    number, delivery date, firmware version, and a fault description of at
    least two sentences. RMA numbers are issued within 4 business hours and
    are valid for 30 days; unused numbers expire automatically.

Q2. HOW LONG DOES RMA EVALUATION TAKE?
A2. Evaluation takes 48 hours from receipt of the unit at our Rotterdam
    facility (Vondelingenweg 601, 3196 KK Rotterdam, The Netherlands). The
    48-hour clock starts when the package is scanned at goods-in, so use a
    trackable carrier. You receive an evaluation report by e-mail stating the
    confirmed fault, the warranty decision, and - for out-of-warranty units -
    a repair quote. Units that arrive without an RMA number are quarantined
    and the evaluation clock does not start until the number is matched.

Q3. HOW LONG DOES THE REPAIR TAKE?
A3. Repairs are completed within 5-7 working days after evaluation for all
    products whose spare parts are stocked in Rotterdam (every model shipped
    since 2019). Safety-certified X300 repairs can take up to 10 working days
    because the SIL2 certificate chain requires a documented final test and
    sign-off by our compliance engineer. Every repair returns with a 90-day
    warranty on the repair itself, independent of the original warranty term.

Q4. WHICH DAMAGES ARE NOT COVERED BY WARRANTY?
A4. The main exclusions are: water damage on devices that are not rated IP68
    (the X100 is IP54 and the X200 is IP65); unauthorized repair or opening
    of sealed housings; overvoltage or lightning events beyond the supply
    tolerance printed in the manual; wear parts beyond their rated life
    (relay contacts 100,000 cycles, RTC batteries 5 years); and damage from
    installation contrary to the manuals. The full list is in the Warranty
    Policy, document DOC-WAR-POL-001, section 4.

Q5. MY UNIT FAILED STRAIGHT AFTER DELIVERY. IS THERE A DOA RULE?
A5. Yes. Units that fail within 14 days of delivery qualify as dead-on-arrival
    (DOA). Report the failure within that window and we ship an advance
    replacement before your defective unit is returned. DOA handling is only
    available for devices bought new from IndustriOS Dynamics or an
    authorised distributor.

Q6. HOW SHOULD I PACKAGE THE UNIT?
A6. Use the original IndustriOS packaging where possible. Otherwise double-
    box the unit with 50 mm cushioning on all sides, put the controller or
    sensor in an ESD bag, and include a printed copy of the RMA confirmation.
    Damage caused by customer packaging is not covered by warranty - see
    question Q4 and Warranty Policy section 4.

Q7. WHO PAYS THE FREIGHT?
A7. In-warranty EU customers receive a prepaid return label by e-mail with
    the RMA number. Customers outside the EU pay outbound freight and are
    reimbursed within 14 days when a warranty defect is confirmed. Out-of-
    warranty customers pay freight in both directions.

Q8. CAN I GET A LOANER UNIT DURING THE REPAIR?
A8. We keep a loaner pool of X200 and X300 controllers for a maximum of 21
    days, subject to availability. Request the loaner in the RMA request.
    The 150 EUR administration fee is waived for customers with a confirmed
    in-warranty repair. Loaner units must be returned with the same firmware
    version they shipped with; your project is restored to your own unit.

Q9. WHAT HAPPENS TO MY DATA AND CONFIGURATION?
A9. Units are wiped to a factory state as part of every repair - this is a
    GDPR requirement and a safety measure. You MUST back up your project,
    parameters, and calibration offsets before shipping; see the Maintenance
    Schedule (DOC-MNT-SC-003, section 4) for the backup procedure. IndustriOS
    Dynamics does not recover customer data from returned units.

Q10. THE REPAIR IS TAKING TOO LONG - HOW DO I ESCALATE?
A10. If five business days pass after evaluation without a status update,
     reply to any RMA e-mail with the word ESCALATE in the subject line. The
     duty manager responds within 4 business hours. The standard escalation
     path is: L1 site technician, then L2 IndustriOS support, then L3
     engineering in Rotterdam (48-72 hours for defect analysis); see the
     Troubleshooting Guide (DOC-TSH-FC-006) for the criteria at each level.

================================================================================
  Contact: support@industrios-dynamics.com | +31 10 555 0142 (Mon-Fri CET)
================================================================================
"""

TROUBLESHOOTING = """
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
"""

DEPLOYMENT_CHECKLIST = """
# Deployment Checklist — X-Series Controllers

**Document ID:** DOC-DEP-CL-004 | **Revision:** 2.1 | **Effective date:** 2024-01-15 | **Classification:** Customer

**Issued by:** IndustriOS Dynamics Ltd., Rotterdam

## 1. Pre-Installation Checklist

- [ ] Site survey completed and cabinet layout drawing approved
- [ ] Ambient temperature at the mounting location verified below 45 °C (24-hour measurement)
- [ ] Clearance of 100 mm on all sides of the controller available and measured
- [ ] DIN rail EN 60715 installed, level, with end clamps fitted
- [ ] 24/48 VDC power budget calculated with 30% headroom
- [ ] Cable glands sized and EMC segregation (300 mm from VFD/motor cables) marked out
- [ ] Network port patched, VLAN 10 confirmed by the network team
- [ ] Spare parts on site: fuses, one air filter, one replacement relay board
- [ ] Firmware package (v3.2.1 or later) and project backup available on the service laptop

## 2. Site Requirements

| Requirement | Limit | Reason |
|---|---|---|
| Ambient temperature | Below 45 °C at cabinet interior | Derating and overtemperature risk (E103) |
| Clearance around controller | 100 mm all sides | Convection cooling per manuals |
| EMC separation from VFD cables | 300 mm | Noise-induced input glitches |
| Supply tolerance | +/- 10% of nominal voltage | Prevents E101/E102 |
| Protective earth continuity | Below 0.1 ohm | Safety per DOC-SAF-CG-002 |
| Vibration | Below 2 g, 10-500 Hz | Connector seating and relay contact life |

## 3. Commissioning Steps (1-9)

1. Mechanical: mount the controller, verify DIN-rail latch engagement and the 100 mm clearance.
2. Grounding: measure protective-earth continuity; the result must be below 0.1 ohm.
3. Power: verify supply voltage and polarity before connecting the controller, then energise and confirm the absence of E101/E102 in the log.
4. Network: set the static IP, subnet, gateway, and VLAN tag per DOC-NET-IG-003; ping the SCADA host from the controller.
5. Firmware: flash the approved release (v3.2.1 or later) after taking the mandatory full backup.
6. Project: restore the project and parameter set, then verify the project checksum matches the engineering record.
7. I/O: run the point-to-point test on 100% of inputs and the forced test on 100% of outputs; record the results.
8. Safety: verify every safety function (E-stop, guards, interlocks) per DOC-SAF-CG-002; on the X300, complete and archive the SIL2 proof test.
9. Documentation: file this signed checklist, register the warranty (X200 only: within 30 days of purchase), and record the firmware version and project checksum.

## 4. Sign-Off

| Field | Entry |
|---|---|
| Installer name and company | ______________ |
| Installer signature / date | ______________ |
| Customer representative | ______________ |
| Customer signature / date | ______________ |
| Commissioning date | ______________ |
| Firmware version installed | ______________ |
| Project checksum (first 16 hex of SHA-256) | ______________ |
| Witness for safety verification (step 8) | ______________ |

## 5. Rejection Criteria

Do not hand the installation over to operations if any of the following applies: any commissioning step 1-8 is incomplete or failed; an error code from the E101-E120 range is active; the ambient temperature at the cabinet exceeds 45 °C without an approved derating; the clearance is below 100 mm; or the X200 warranty registration window (30 days from purchase) has fewer than 5 days remaining without registration. A rejected installation must be re-inspected after remediation using a fresh copy of this checklist.
"""

# Mapping: filename -> content  (13 documents: 12 markdown + 1 plain text)
TEXT_DOCS = {
    "x100_controller_manual.md": X100_MANUAL,
    "x200_controller_manual.md": X200_MANUAL,
    "x300_controller_manual.md": X300_MANUAL,
    "s200_sensor_datasheet.md": S200_DATASHEET,
    "s350_sensor_datasheet.md": S350_DATASHEET,
    "warranty_policy.md": WARRANTY_POLICY,
    "safety_compliance_guide.md": SAFETY_GUIDE,
    "maintenance_schedule.md": MAINTENANCE_SCHEDULE,
    "firmware_release_notes.md": FIRMWARE_NOTES,
    "network_integration_guide.md": NETWORK_GUIDE,
    "rma_faq.txt": RMA_FAQ,
    "troubleshooting_flowchart.md": TROUBLESHOOTING,
    "deployment_checklist.md": DEPLOYMENT_CHECKLIST,
}

# ============================================================================
#  CHART ASSET (matplotlib)
# ============================================================================

def build_warranty_chart(out_path: Path) -> Path:
    """Render the 'warranty duration by model' bar chart PNG (low-saturation palette)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = ["X100\n(IP54)", "X200\n(IP65)", "X300\n(IP67)", "S200/S350\n(IP68)"]
    standard = [12, 24, 36, 18]      # months, standard warranty
    extension = [0, 12, 0, 0]        # extra months (X200 only, with registration)

    fig, ax = plt.subplots(figsize=(7.6, 3.8), dpi=160)
    ax.bar(models, standard, width=0.55, color="#64748b",
           label="Standard warranty")
    ax.bar(models, extension, width=0.55, bottom=standard, color="#059669",
           label="Extended warranty (X200, registration within 30 days)")

    for i, (s, e) in enumerate(zip(standard, extension)):
        ax.text(i, s + e + 1.0, "%d mo" % (s + e), ha="center", va="bottom",
                fontsize=9, color="#0F172A")
    # label the green extension segment of the X200
    ax.text(1, 30, "+12", ha="center", va="center", fontsize=8, color="white")

    ax.set_ylabel("Coverage (months)")
    ax.set_title("IndustriOS Dynamics - Warranty Duration by Product", fontsize=11, pad=10)
    ax.set_ylim(0, 44)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    return out_path


# ============================================================================
#  PDF BUILDING (reportlab)
# ============================================================================

# Characters outside reportlab's WinAnsi encoding, mapped to safe equivalents.
_CHAR_FIXES = {
    "\u2212": "-", "\u2265": ">=", "\u2264": "<=", "\u03a9": "Ohm",
    "\u2126": "Ohm", "\u2011": "-", "\u00a0": " ", "\u2192": "->",
}


def _fix_chars(text: str) -> str:
    for bad, good in _CHAR_FIXES.items():
        text = text.replace(bad, good)
    return text


def _inline(text: str) -> str:
    """Escape for reportlab Paragraphs and apply **bold** markers."""
    from xml.sax.saxutils import escape
    t = escape(_fix_chars(text))
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", t)
    return t


def _pdf_styles() -> dict:
    from reportlab.lib.colors import HexColor
    from reportlab.lib.styles import ParagraphStyle

    ink = HexColor("#0F172A")
    slate = HexColor("#475569")
    return {
        "Meta": ParagraphStyle("Meta", fontName="Helvetica-Oblique", fontSize=8,
                               leading=11, textColor=slate, spaceAfter=10),
        "Title": ParagraphStyle("TitleX", fontName="Helvetica-Bold", fontSize=17,
                                leading=21, textColor=ink, spaceAfter=6),
        "H2": ParagraphStyle("H2X", fontName="Helvetica-Bold", fontSize=12.5,
                             leading=16, textColor=ink, spaceBefore=4, spaceAfter=4),
        "H3": ParagraphStyle("H3X", fontName="Helvetica-Bold", fontSize=10.5,
                             leading=13.5, textColor=slate, spaceBefore=6, spaceAfter=3),
        "Body": ParagraphStyle("Body", fontName="Helvetica", fontSize=9.5,
                               leading=13.2, textColor=ink, spaceAfter=5),
        "Bullet": ParagraphStyle("Bullet", fontName="Helvetica", fontSize=9.5,
                                 leading=13.2, textColor=ink, leftIndent=12,
                                 bulletIndent=2, spaceAfter=2),
        "Step": ParagraphStyle("Step", fontName="Helvetica", fontSize=9.5,
                               leading=13.2, textColor=ink, leftIndent=10, spaceAfter=2),
        "Cell": ParagraphStyle("Cell", fontName="Helvetica", fontSize=8.2,
                               leading=10.8, textColor=ink),
        "CellHead": ParagraphStyle("CellHead", fontName="Helvetica-Bold", fontSize=8.2,
                                   leading=10.8, textColor=ink),
    }


def _split_table_row(line: str) -> list:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _is_separator_row(cells: list) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c != "")


def _build_table_flowable(lines: list, styles: dict, avail: float) -> list:
    from reportlab.lib.colors import HexColor, white
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

    header = _split_table_row(lines[0])
    rows = []
    for ln in lines[1:]:
        cells = _split_table_row(ln)
        if _is_separator_row(cells):
            continue
        rows.append(cells)

    ncols = max([len(header)] + [len(r) for r in rows]) if rows else len(header)
    header = header + [""] * (ncols - len(header))
    rows = [r + [""] * (ncols - len(r)) for r in rows]

    def cell(txt, is_header):
        return Paragraph(_inline(txt),
                         styles["CellHead"] if is_header else styles["Cell"])

    data = [[cell(c, True) for c in header]]
    for r in rows:
        data.append([cell(c, False) for c in r])

    if ncols == 2:
        widths = [avail * 0.33, avail * 0.67]
    elif ncols == 3:
        widths = [avail * 0.17, avail * 0.415, avail * 0.415]
    else:
        widths = [avail / ncols] * ncols

    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#F8FAFC")]),
        ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#94A3B8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return [Spacer(1, 2), table, Spacer(1, 8)]


def _md_to_story(md_text: str, styles: dict, image_base: Path, avail: float) -> list:
    """Convert markdown to reportlab flowables.

    Layout rule: every H2 section starts on a new page (except the first one,
    which follows the title block on page 1) so that 'page N' citations are
    meaningful.
    """
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import (HRFlowable, Image as RLImage, PageBreak,
                                    Paragraph, Spacer)

    story = []
    table_buf: list = []
    first_h2_seen = False

    def flush_table():
        nonlocal table_buf
        if table_buf:
            story.extend(_build_table_flowable(table_buf, styles, avail))
            table_buf = []

    for raw in md_text.splitlines():
        line = raw.rstrip()
        s = line.strip()

        if s.startswith("|"):
            table_buf.append(s)
            continue
        flush_table()

        if not s:
            continue

        m = re.match(r"^(#{1,4})\s+(.+)$", s)
        if m:
            level, text = len(m.group(1)), m.group(2)
            if level == 1:
                story.append(Paragraph(_inline(text), styles["Title"]))
                story.append(Spacer(1, 4))
            elif level == 2:
                if first_h2_seen:
                    story.append(PageBreak())          # section = own page
                first_h2_seen = True
                story.append(Paragraph(_inline(text), styles["H2"]))
                story.append(Spacer(1, 4))
            else:
                story.append(Paragraph(_inline(text), styles["H3"]))
                story.append(Spacer(1, 3))
            continue

        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", s)
        if m:  # markdown image -> embedded PNG
            img_path = image_base / m.group(2)
            if img_path.exists():
                iw, ih = ImageReader(str(img_path)).getSize()
                target_w = min(440.0, avail)
                target_h = ih * target_w / iw
                story.append(Spacer(1, 4))
                story.append(RLImage(str(img_path), width=target_w, height=target_h))
                story.append(Spacer(1, 8))
            continue

        m = re.match(r"^[-*]\s+(.+)$", s)
        if m:  # bullet item (incl. "- [ ]" checkboxes)
            story.append(Paragraph(_inline(m.group(1)), styles["Bullet"],
                                   bulletText="-"))
            story.append(Spacer(1, 1))
            continue

        m = re.match(r"^(\d+)[.)]\s+(.+)$", s)
        if m:  # numbered step — keep the number, it is meaningful content
            story.append(Paragraph("%s. %s" % (m.group(1), _inline(m.group(2))),
                                   styles["Step"]))
            continue

        if re.fullmatch(r"-{3,}", s):  # horizontal rule
            story.append(HRFlowable(width="100%", thickness=0.6,
                                    color=HexColorSafe("#CBD5E1"),
                                    spaceBefore=4, spaceAfter=6))
            continue

        story.append(Paragraph(_inline(s), styles["Body"]))
        story.append(Spacer(1, 3))

    flush_table()
    return story


def HexColorSafe(hexstr: str):
    from reportlab.lib.colors import HexColor
    return HexColor(hexstr)


def _on_page_factory(header_title: str):
    from reportlab.lib.colors import HexColor
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm

    def _on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(HexColor("#64748b"))
        if doc.page > 1:  # running header from page 2 on
            canvas.drawString(20 * mm, A4[1] - 13 * mm, header_title)
            canvas.setStrokeColor(HexColor("#CBD5E1"))
            canvas.setLineWidth(0.5)
            canvas.line(20 * mm, A4[1] - 15 * mm, A4[0] - 20 * mm, A4[1] - 15 * mm)
        canvas.drawString(20 * mm, 11 * mm,
                          "IndustriOS Dynamics Ltd. - Technical Documentation, Rotterdam")
        canvas.drawRightString(A4[0] - 20 * mm, 11 * mm, "Page %d" % doc.page)
        canvas.restoreState()

    return _on_page


def build_pdf(md_text: str, md_filename: str, out_path: Path, header_title: str) -> Path:
    """Typeset the markdown source into a paginated PDF with real page numbers."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate

    styles = _pdf_styles()
    avail = A4[0] - 40 * mm

    story = [Paragraph(
        _inline("PDF edition of %s - content identical to the markdown source."
                % md_filename), styles["Meta"])]
    story.extend(_md_to_story(md_text.strip(), styles, CORPUS_DIR, avail))

    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm, topMargin=22 * mm, bottomMargin=20 * mm,
        title=header_title, author="IndustriOS Dynamics Ltd.",
        subject="Demo knowledge-base corpus document",
    )
    on_page = _on_page_factory(header_title)
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return out_path


PDF_JOBS = [
    # (md constant, md filename, pdf filename, running-header title)
    (X200_MANUAL, "x200_controller_manual.md", "x200_manual.pdf",
     "X200 Industrial Controller - User Manual (DOC-X200-UM-001)"),
    (WARRANTY_POLICY, "warranty_policy.md", "warranty_policy.pdf",
     "Warranty Policy - X-Series and S-Series (DOC-WAR-POL-001)"),
    (FIRMWARE_NOTES, "firmware_release_notes.md", "firmware_release_notes.pdf",
     "Firmware Release Notes - X-Series Controllers (DOC-FW-RN-005)"),
]


# ============================================================================
#  MAIN
# ============================================================================

def _print_summary() -> None:
    print("\n" + "=" * 78)
    print("CORPUS VERIFICATION SUMMARY  (%s)" % CORPUS_DIR)
    print("=" * 78)

    md_txt = pdf = png = 0
    for path in sorted(CORPUS_DIR.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(CORPUS_DIR))

        if path.suffix in (".md", ".txt"):
            words = len(re.findall(r"\S+", path.read_text(encoding="utf-8")))
            md_txt += 1
            flag = "" if 500 <= words <= 900 else "   <-- OUT OF 500-900 RANGE"
            print("  %-34s %5d words%s" % (rel, words, flag))

        elif path.suffix == ".pdf":
            pdf += 1
            try:
                import pdfplumber
                with pdfplumber.open(str(path)) as doc:
                    pages = len(doc.pages)
                    extractable = all((p.extract_text() or "").strip() for p in doc.pages)
                    sample = (doc.pages[0].extract_text() or "")[:60].replace("\n", " ")
                flag = "" if pages >= 3 else "   <-- FEWER THAN 3 PAGES"
                print("  %-34s %2d pages, text-extractable=%s%s"
                      % (rel, pages, extractable, flag))
                print("       page-1 text: %r" % sample)
            except ImportError:
                print("  %-34s (pdfplumber not installed - not verified)" % rel)

        else:
            png += 1
            print("  %-34s %7d bytes (chart asset)"
                  % (rel, path.stat().st_size))

    print("-" * 78)
    print("  Total: %d md/txt + %d pdf + %d png = %d files"
          % (md_txt, pdf, png, md_txt + pdf + png))
    print("=" * 78)


def main() -> None:
    print("IndustriOS Dynamics - demo corpus generator")
    print("Corpus root: %s" % CORPUS_DIR)

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # -- 1. markdown / plain-text documents -----------------------------------
    print("[1/3] Writing %d markdown/text documents ..." % len(TEXT_DOCS))
    for name, content in TEXT_DOCS.items():
        (CORPUS_DIR / name).write_text(content.strip() + "\n", encoding="utf-8")
        print("      wrote %s" % name)

    # -- 2. chart asset -------------------------------------------------------
    print("[2/3] Rendering chart asset ...")
    chart = build_warranty_chart(ASSETS_DIR / "warranty_duration.png")
    print("      wrote %s (%d bytes)" % (chart, chart.stat().st_size))

    # -- 3. PDFs ---------------------------------------------------------------
    print("[3/3] Building %d typeset PDFs ..." % len(PDF_JOBS))
    for md_text, md_name, pdf_name, title in PDF_JOBS:
        out = build_pdf(md_text, md_name, CORPUS_DIR / pdf_name, title)
        print("      wrote %s (from %s)" % (out.name, md_name))

    _print_summary()


if __name__ == "__main__":
    main()
