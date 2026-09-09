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
