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
