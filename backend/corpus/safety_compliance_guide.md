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
