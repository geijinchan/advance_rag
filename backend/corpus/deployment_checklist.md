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
