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
