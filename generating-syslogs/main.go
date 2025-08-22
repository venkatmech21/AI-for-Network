package main

import (
	"bufio"
	"fmt"
	"log"
	"math/rand"
	"net"
	"os"
	"strconv"
	"strings"
	"time"
)

// SyslogMessage represents a structured syslog message
type SyslogMessage struct {
	Timestamp string
	Hostname  string
	Facility  int
	Severity  int
	Program   string
	PID       int
	Message   string
}

// DeviceType represents different network device types
type DeviceType int

const (
	CiscoIOSXE DeviceType = iota
	PaloAltoFirewall
	JuniperRouter
)

// EventType represents different types of network events
type EventType int

const (
	LoginEvent EventType = iota
	BGPFlapEvent
	OSPFFlapEvent
	LatencyEvent
	SecurityEvent
	TCPAttackEvent
	InterfaceEvent
	SystemEvent
	RoutingEvent
	VPNEvent
)

// SyslogGenerator handles generation of syslog messages
type SyslogGenerator struct {
	deviceNames   map[DeviceType][]string
	ipRanges      []string
	userNames     []string
	attackTypes   []string
	countries     []string
	asNumbers     []int
	interfaces    map[DeviceType][]string
	bgpPeers      []string
	ospfAreas     []string
	protocols     []string
	ports         []int
	rand          *rand.Rand
}

// NewSyslogGenerator creates a new syslog generator with predefined data
func NewSyslogGenerator() *SyslogGenerator {
	source := rand.NewSource(time.Now().UnixNano())
	r := rand.New(source)

	return &SyslogGenerator{
		deviceNames: map[DeviceType][]string{
			CiscoIOSXE: {
				"CORE-RTR-01", "CORE-RTR-02", "EDGE-RTR-01", "EDGE-RTR-02",
				"DIST-SW-01", "DIST-SW-02", "ACCESS-SW-01", "ACCESS-SW-02",
				"WAN-RTR-01", "WAN-RTR-02", "MPLS-PE-01", "MPLS-PE-02",
			},
			PaloAltoFirewall: {
				"PA-FW-01", "PA-FW-02", "PA-FW-DMZ-01", "PA-FW-DMZ-02",
				"PA-FW-INET-01", "PA-FW-INET-02", "PA-FW-DC-01", "PA-FW-DC-02",
				"PA-FW-BRANCH-01", "PA-FW-BRANCH-02", "PA-FW-MGMT-01", "PA-FW-MGMT-02",
			},
			JuniperRouter: {
				"JUN-RTR-CORE-01", "JUN-RTR-CORE-02", "JUN-RTR-EDGE-01", "JUN-RTR-EDGE-02",
				"JUN-MX-01", "JUN-MX-02", "JUN-EX-01", "JUN-EX-02",
				"JUN-SRX-01", "JUN-SRX-02", "JUN-QFX-01", "JUN-QFX-02",
			},
		},
		ipRanges: []string{
			"192.168.1.", "192.168.10.", "192.168.100.", "10.0.1.", "10.0.10.",
			"10.10.1.", "172.16.1.", "172.16.10.", "172.31.1.", "203.0.113.",
			"198.51.100.", "8.8.8.", "1.1.1.", "4.4.4.", "9.9.9.",
		},
		userNames: []string{
			"admin", "operator", "netadmin", "security", "monitor", "backup",
			"john.doe", "jane.smith", "mike.wilson", "sarah.jones", "david.brown",
			"attacker", "scanner", "botnet", "malware", "hacker",
		},
		attackTypes: []string{
			"SYN flood", "UDP flood", "ICMP flood", "HTTP flood", "DNS amplification",
			"port scan", "vulnerability scan", "brute force", "DDoS", "botnet C&C",
			"malware communication", "data exfiltration", "lateral movement",
		},
		countries: []string{
			"US", "CN", "RU", "BR", "IN", "DE", "FR", "GB", "JP", "CA",
			"AU", "KR", "MX", "IT", "ES", "NL", "TR", "PL", "AR", "TH",
		},
		asNumbers: []int{
			64512, 64513, 64514, 64515, 64516, 64517, 64518, 64519, 64520,
			7018, 1299, 3356, 174, 2914, 3257, 6453, 1273, 5511, 701,
		},
		interfaces: map[DeviceType][]string{
			CiscoIOSXE: {
				"GigabitEthernet0/0/0", "GigabitEthernet0/0/1", "GigabitEthernet0/1/0",
				"TenGigabitEthernet0/0/0", "TenGigabitEthernet0/0/1", "Serial0/0/0",
				"Loopback0", "Loopback1", "Tunnel0", "Tunnel1", "Vlan1", "Vlan10",
			},
			PaloAltoFirewall: {
				"ethernet1/1", "ethernet1/2", "ethernet1/3", "ethernet1/4",
				"ethernet1/5", "ethernet1/6", "ae1", "ae2", "tunnel.1", "tunnel.2",
				"vlan.10", "vlan.20", "vlan.100", "management",
			},
			JuniperRouter: {
				"ge-0/0/0", "ge-0/0/1", "ge-0/1/0", "xe-0/0/0", "xe-0/0/1",
				"et-0/0/0", "lo0", "lo0.0", "st0.1", "st0.2", "irb.10", "irb.20",
			},
		},
		bgpPeers: []string{
			"203.0.113.1", "203.0.113.2", "198.51.100.1", "198.51.100.2",
			"8.8.8.8", "1.1.1.1", "4.4.4.4", "9.9.9.9", "208.67.222.222",
		},
		ospfAreas: []string{
			"0.0.0.0", "0.0.0.1", "0.0.0.2", "0.0.0.10", "0.0.0.100",
		},
		protocols: []string{
			"TCP", "UDP", "ICMP", "ESP", "AH", "GRE", "OSPF", "BGP", "EIGRP",
		},
		ports: []int{
			22, 23, 25, 53, 80, 110, 143, 443, 993, 995, 1433, 3389, 5900,
			8080, 8443, 3306, 5432, 6379, 27017, 9200, 5601, 9300,
		},
		rand: r,
	}
}

// generateRandomIP generates a random IP address
func (sg *SyslogGenerator) generateRandomIP() string {
	prefix := sg.ipRanges[sg.rand.Intn(len(sg.ipRanges))]
	suffix := sg.rand.Intn(255) + 1
	return prefix + strconv.Itoa(suffix)
}

// generateTimestamp generates a timestamp within the last 24 hours
func (sg *SyslogGenerator) generateTimestamp() string {
	now := time.Now()
	// Generate timestamp within last 24 hours
	randomSeconds := sg.rand.Intn(24 * 60 * 60)
	timestamp := now.Add(-time.Duration(randomSeconds) * time.Second)
	return timestamp.Format("Jan 02 15:04:05")
}

// generateCiscoIOSXEMessage generates Cisco IOS-XE syslog messages
func (sg *SyslogGenerator) generateCiscoIOSXEMessage(eventType EventType, hostname string) SyslogMessage {
	timestamp := sg.generateTimestamp()
	facility := 16 // local0
	severity := sg.rand.Intn(7) + 1
	
	var program, message string
	
	switch eventType {
	case LoginEvent:
		if sg.rand.Float32() < 0.7 {
			// Successful login
			severity = 6 // info
			program = "SSH2"
			user := sg.userNames[sg.rand.Intn(len(sg.userNames))]
			srcIP := sg.generateRandomIP()
			message = fmt.Sprintf("SSH2_SESSION: SSH2 Session request from %s (tty = 0) using crypto cipher 'aes256-ctr', hmac 'hmac-sha2-256' Succeeded for user '%s'", srcIP, user)
		} else {
			// Failed login
			severity = 4 // warning
			program = "SSH2"
			user := sg.userNames[sg.rand.Intn(len(sg.userNames))]
			srcIP := sg.generateRandomIP()
			message = fmt.Sprintf("SSH2_SESSION: SSH2 Session request from %s (tty = 0) using crypto cipher 'aes256-ctr', hmac 'hmac-sha2-256' Failed for user '%s'", srcIP, user)
		}
		
	case BGPFlapEvent:
		severity = 3 // error
		program = "BGP"
		peer := sg.bgpPeers[sg.rand.Intn(len(sg.bgpPeers))]
		asNum := sg.asNumbers[sg.rand.Intn(len(sg.asNumbers))]
		if sg.rand.Float32() < 0.5 {
			message = fmt.Sprintf("BGP-3-NOTIFICATION_SENT: sent to neighbor %s (AS%d) 4/0 (hold time expired) 0 bytes", peer, asNum)
		} else {
			message = fmt.Sprintf("BGP-5-ADJCHANGE: neighbor %s (AS%d) Up", peer, asNum)
		}
		
	case OSPFFlapEvent:
		severity = 4 // warning
		program = "OSPF"
		//area := sg.ospfAreas[sg.rand.Intn(len(sg.ospfAreas))]
		routerID := sg.generateRandomIP()
		if sg.rand.Float32() < 0.5 {
			message = fmt.Sprintf("OSPF-5-ADJCHG: Process 1, Nbr %s on GigabitEthernet0/0/0 from FULL to DOWN, Neighbor Down: Dead timer expired", routerID)
		} else {
			message = fmt.Sprintf("OSPF-5-ADJCHG: Process 1, Nbr %s on GigabitEthernet0/0/0 from LOADING to FULL, Loading Done", routerID)
		}
		
	case LatencyEvent:
		severity = 4 // warning
		program = "IPSLA"
		target := sg.generateRandomIP()
		latency := sg.rand.Intn(500) + 100
		message = fmt.Sprintf("IPSLA-6-PROBE_LATENCY: RTT probe to %s exceeded threshold: %dms (threshold: 100ms)", target, latency)
		
	case SecurityEvent:
		severity = 2 // critical
		program = "SEC_LOGIN"
		srcIP := sg.generateRandomIP()
		attempts := sg.rand.Intn(10) + 5
		message = fmt.Sprintf("SEC_LOGIN-4-LOGIN_FAILED: Login failed for user from %s, %d failed attempts", srcIP, attempts)
		
	case TCPAttackEvent:
		severity = 1 // alert
		program = "FIREWALL"
		srcIP := sg.generateRandomIP()
		dstIP := sg.generateRandomIP()
		port := sg.ports[sg.rand.Intn(len(sg.ports))]
		attackType := sg.attackTypes[sg.rand.Intn(len(sg.attackTypes))]
		message = fmt.Sprintf("FIREWALL-1-ATTACK_DETECTED: %s detected from %s to %s:%d", attackType, srcIP, dstIP, port)
		
	case InterfaceEvent:
		severity = 3 // error
		program = "LINEPROTO"
		iface := sg.interfaces[CiscoIOSXE][sg.rand.Intn(len(sg.interfaces[CiscoIOSXE]))]
		if sg.rand.Float32() < 0.3 {
			message = fmt.Sprintf("LINEPROTO-5-UPDOWN: Line protocol on Interface %s, changed state to down", iface)
		} else {
			message = fmt.Sprintf("LINEPROTO-5-UPDOWN: Line protocol on Interface %s, changed state to up", iface)
		}
		
	case SystemEvent:
		severity = 6 // info
		program = "SYS"
		if sg.rand.Float32() < 0.5 {
			cpu := sg.rand.Intn(40) + 60
			message = fmt.Sprintf("SYS-6-CPU_USAGE: CPU utilization for 5 seconds: %d%%; one minute: %d%%; five minutes: %d%%", cpu, cpu-5, cpu-10)
		} else {
			memory := sg.rand.Intn(30) + 70
			message = fmt.Sprintf("SYS-6-MEM_USAGE: Memory utilization: %d%% used", memory)
		}
		
	case RoutingEvent:
		severity = 5 // notice
		program = "ROUTING"
		network := fmt.Sprintf("%s0/24", sg.ipRanges[sg.rand.Intn(len(sg.ipRanges))])
		nextHop := sg.generateRandomIP()
		message = fmt.Sprintf("ROUTING-5-ROUTE_CHANGE: Route %s via %s added to routing table", network, nextHop)
		
	default:
		severity = 6
		program = "SYSTEM"
		message = "Unknown event type"
	}
	
	return SyslogMessage{
		Timestamp: timestamp,
		Hostname:  hostname,
		Facility:  facility,
		Severity:  severity,
		Program:   program,
		PID:       sg.rand.Intn(30000) + 1000,
		Message:   message,
	}
}

// generatePaloAltoMessage generates Palo Alto firewall syslog messages
func (sg *SyslogGenerator) generatePaloAltoMessage(eventType EventType, hostname string) SyslogMessage {
	timestamp := sg.generateTimestamp()
	facility := 16 // local0
	severity := sg.rand.Intn(7) + 1
	
	var program, message string
	
	switch eventType {
	case LoginEvent:
		if sg.rand.Float32() < 0.8 {
			// Successful login
			severity = 6 // info
			program = "mgmtsrvr"
			user := sg.userNames[sg.rand.Intn(len(sg.userNames))]
			srcIP := sg.generateRandomIP()
			message = fmt.Sprintf("Successful admin authentication for user '%s' from %s", user, srcIP)
		} else {
			// Failed login
			severity = 4 // warning
			program = "mgmtsrvr"
			user := sg.userNames[sg.rand.Intn(len(sg.userNames))]
			srcIP := sg.generateRandomIP()
			message = fmt.Sprintf("Authentication failed for user '%s' from %s", user, srcIP)
		}
		
	case SecurityEvent:
		severity = 2 // critical
		program = "threat"
		srcIP := sg.generateRandomIP()
		dstIP := sg.generateRandomIP()
		srcPort := sg.ports[sg.rand.Intn(len(sg.ports))]
		dstPort := sg.ports[sg.rand.Intn(len(sg.ports))]
		threatName := []string{"Malware.Generic", "Trojan.Win32", "Backdoor.Linux", "Exploit.CVE-2021", "Botnet.C2"}[sg.rand.Intn(5)]
		message = fmt.Sprintf("THREAT,1,2023/08/16 %s,%s,THREAT,%s,%s,%s,%s,%d,%d,0x0,tcp,alert,\"%s\"",
			sg.generateTimestamp(), hostname, srcIP, dstIP, srcIP, dstIP, srcPort, dstPort, threatName)
		
	case TCPAttackEvent:
		severity = 1 // alert
		program = "threat"
		srcIP := sg.generateRandomIP()
		dstIP := sg.generateRandomIP()
		srcPort := sg.rand.Intn(65535) + 1
		dstPort := sg.ports[sg.rand.Intn(len(sg.ports))]
		attackType := sg.attackTypes[sg.rand.Intn(len(sg.attackTypes))]
		message = fmt.Sprintf("THREAT,1,2023/08/16 %s,%s,THREAT,%s,%s,%s,%s,%d,%d,0x0,tcp,alert,\"%s detected\"",
			sg.generateTimestamp(), hostname, srcIP, dstIP, srcIP, dstIP, srcPort, dstPort, attackType)
		
	case InterfaceEvent:
		severity = 4 // warning
		program = "devsrvr"
		iface := sg.interfaces[PaloAltoFirewall][sg.rand.Intn(len(sg.interfaces[PaloAltoFirewall]))]
		if sg.rand.Float32() < 0.3 {
			message = fmt.Sprintf("Interface %s link state changed to down", iface)
		} else {
			message = fmt.Sprintf("Interface %s link state changed to up", iface)
		}
		
	case SystemEvent:
		severity = 6 // info
		program = "devsrvr"
		if sg.rand.Float32() < 0.3 {
			cpu := sg.rand.Intn(40) + 60
			message = fmt.Sprintf("High CPU usage detected: %d%% (threshold: 80%%)", cpu)
		} else if sg.rand.Float32() < 0.6 {
			memory := sg.rand.Intn(30) + 70
			message = fmt.Sprintf("Memory usage: %d%% of total memory", memory)
		} else {
			disk := sg.rand.Intn(20) + 80
			message = fmt.Sprintf("Disk usage warning: /opt/pancfg at %d%% capacity", disk)
		}
		
	case VPNEvent:
		severity = 6 // info
		program = "sslmgr"
		if sg.rand.Float32() < 0.7 {
			user := sg.userNames[sg.rand.Intn(len(sg.userNames))]
			srcIP := sg.generateRandomIP()
			message = fmt.Sprintf("GlobalProtect gateway user authentication succeeded. User: %s, Source IP: %s", user, srcIP)
		} else {
			user := sg.userNames[sg.rand.Intn(len(sg.userNames))]
			srcIP := sg.generateRandomIP()
			message = fmt.Sprintf("GlobalProtect gateway user authentication failed. User: %s, Source IP: %s", user, srcIP)
		}
		
	case LatencyEvent:
		severity = 4 // warning
		program = "reportd"
		target := sg.generateRandomIP()
		latency := sg.rand.Intn(300) + 100
		message = fmt.Sprintf("Health monitor probe to %s failed - response time %dms exceeds threshold", target, latency)
		
	default:
		severity = 6
		program = "system"
		message = "Unknown event type"
	}
	
	return SyslogMessage{
		Timestamp: timestamp,
		Hostname:  hostname,
		Facility:  facility,
		Severity:  severity,
		Program:   program,
		PID:       sg.rand.Intn(30000) + 1000,
		Message:   message,
	}
}

// generateJuniperMessage generates Juniper router syslog messages
func (sg *SyslogGenerator) generateJuniperMessage(eventType EventType, hostname string) SyslogMessage {
	timestamp := sg.generateTimestamp()
	facility := 16 // local0
	severity := sg.rand.Intn(7) + 1
	
	var program, message string
	
	switch eventType {
	case LoginEvent:
		if sg.rand.Float32() < 0.75 {
			// Successful login
			severity = 6 // info
			program = "sshd"
			user := sg.userNames[sg.rand.Intn(len(sg.userNames))]
			srcIP := sg.generateRandomIP()
			pid := sg.rand.Intn(30000) + 1000
			message = fmt.Sprintf("sshd[%d]: Accepted password for %s from %s port 22 ssh2", pid, user, srcIP)
		} else {
			// Failed login
			severity = 4 // warning
			program = "sshd"
			user := sg.userNames[sg.rand.Intn(len(sg.userNames))]
			srcIP := sg.generateRandomIP()
			pid := sg.rand.Intn(30000) + 1000
			message = fmt.Sprintf("sshd[%d]: Failed password for %s from %s port 22 ssh2", pid, user, srcIP)
		}
		
	case BGPFlapEvent:
		severity = 3 // error
		program = "rpd"
		peer := sg.bgpPeers[sg.rand.Intn(len(sg.bgpPeers))]
		asNum := sg.asNumbers[sg.rand.Intn(len(sg.asNumbers))]
		if sg.rand.Float32() < 0.5 {
			message = fmt.Sprintf("bgp_peer_err: BGP peer %s (External AS %d): Hold Timer Expired Error", peer, asNum)
		} else {
			message = fmt.Sprintf("bgp_peer_up: BGP peer %s (External AS %d) transitioned from Connect to Established", peer, asNum)
		}
		
	case OSPFFlapEvent:
		severity = 4 // warning
		program = "rpd"
		area := sg.ospfAreas[sg.rand.Intn(len(sg.ospfAreas))]
		neighborID := sg.generateRandomIP()
		iface := sg.interfaces[JuniperRouter][sg.rand.Intn(len(sg.interfaces[JuniperRouter]))]
		if sg.rand.Float32() < 0.5 {
			message = fmt.Sprintf("ospf_neighbor_down: OSPF neighbor %s (area %s) state changed from Full to Down on interface %s due to HelloTimerExpired", neighborID, area, iface)
		} else {
			message = fmt.Sprintf("ospf_neighbor_up: OSPF neighbor %s (area %s) state changed from Init to Full on interface %s", neighborID, area, iface)
		}
		
	case InterfaceEvent:
		severity = 3 // error
		program = "kernel"
		iface := sg.interfaces[JuniperRouter][sg.rand.Intn(len(sg.interfaces[JuniperRouter]))]
		if sg.rand.Float32() < 0.3 {
			message = fmt.Sprintf("%s: link is Down", iface)
		} else {
			message = fmt.Sprintf("%s: link is Up", iface)
		}
		
	case LatencyEvent:
		severity = 4 // warning
		program = "rpd"
		target := sg.generateRandomIP()
		latency := sg.rand.Intn(400) + 150
		message = fmt.Sprintf("ping_probe_failed: Ping probe to %s failed - RTT %dms exceeds threshold 100ms", target, latency)
		
	case SecurityEvent:
		severity = 2 // critical
		program = "kernel"
		srcIP := sg.generateRandomIP()
		dstIP := sg.generateRandomIP()
		port := sg.ports[sg.rand.Intn(len(sg.ports))]
		message = fmt.Sprintf("Security: Suspicious activity detected from %s scanning %s port %d", srcIP, dstIP, port)
		
	case RoutingEvent:
		severity = 5 // notice
		program = "rpd"
		prefix := fmt.Sprintf("%s0/24", sg.ipRanges[sg.rand.Intn(len(sg.ipRanges))])
		nextHop := sg.generateRandomIP()
		protocol := []string{"BGP", "OSPF", "STATIC", "DIRECT"}[sg.rand.Intn(4)]
		message = fmt.Sprintf("route_change: Route %s nexthop %s protocol %s added to inet.0", prefix, nextHop, protocol)
		
	case SystemEvent:
		severity = 6 // info
		program = "chassisd"
		if sg.rand.Float32() < 0.4 {
			temp := sg.rand.Intn(30) + 45
			message = fmt.Sprintf("Temperature alarm: Routing Engine temperature %d°C exceeds minor threshold", temp)
		} else if sg.rand.Float32() < 0.7 {
			cpu := sg.rand.Intn(40) + 60
			message = fmt.Sprintf("CPU utilization: %d%% (5-minute average)", cpu)
		} else {
			memory := sg.rand.Intn(30) + 70
			message = fmt.Sprintf("Memory utilization: %d%% of available memory", memory)
		}
		
	default:
		severity = 6
		program = "kernel"
		message = "Unknown event type"
	}
	
	return SyslogMessage{
		Timestamp: timestamp,
		Hostname:  hostname,
		Facility:  facility,
		Severity:  severity,
		Program:   program,
		PID:       sg.rand.Intn(30000) + 1000,
		Message:   message,
	}
}

// FormatRFC3164 formats the syslog message in RFC3164 format
func (msg *SyslogMessage) FormatRFC3164() string {
	priority := msg.Facility*8 + msg.Severity
	return fmt.Sprintf("<%d>%s %s %s[%d]: %s",
		priority, msg.Timestamp, msg.Hostname, msg.Program, msg.PID, msg.Message)
}

// FormatRFC5424 formats the syslog message in RFC5424 format
func (msg *SyslogMessage) FormatRFC5424() string {
	priority := msg.Facility*8 + msg.Severity
	timestamp := time.Now().Format("2006-01-02T15:04:05.000Z")
	return fmt.Sprintf("<%d>1 %s %s %s %d - - %s",
		priority, timestamp, msg.Hostname, msg.Program, msg.PID, msg.Message)
}

// generateMessages generates specified number of messages for a device type
func (sg *SyslogGenerator) generateMessages(deviceType DeviceType, count int) []SyslogMessage {
	var messages []SyslogMessage
	deviceNames := sg.deviceNames[deviceType]
	
	// Define event distribution (percentages)
	eventDistribution := map[EventType]float32{
		LoginEvent:     0.15, // 15%
		BGPFlapEvent:   0.05, // 5%
		OSPFFlapEvent:  0.05, // 5%
		LatencyEvent:   0.10, // 10%
		SecurityEvent:  0.15, // 15%
		TCPAttackEvent: 0.08, // 8%
		InterfaceEvent: 0.12, // 12%
		SystemEvent:    0.20, // 20%
		RoutingEvent:   0.07, // 7%
		VPNEvent:       0.03, // 3%
	}
	
	for i := 0; i < count; i++ {
		// Select random device
		hostname := deviceNames[sg.rand.Intn(len(deviceNames))]
		
		// Select event type based on distribution
		randVal := sg.rand.Float32()
		var selectedEvent EventType
		cumulative := float32(0)
		
		for eventType, probability := range eventDistribution {
			cumulative += probability
			if randVal <= cumulative {
				selectedEvent = eventType
				break
			}
		}
		
		// Generate message based on device type
		var msg SyslogMessage
		switch deviceType {
		case CiscoIOSXE:
			msg = sg.generateCiscoIOSXEMessage(selectedEvent, hostname)
		case PaloAltoFirewall:
			msg = sg.generatePaloAltoMessage(selectedEvent, hostname)
		case JuniperRouter:
			msg = sg.generateJuniperMessage(selectedEvent, hostname)
		}
		
		messages = append(messages, msg)
	}
	
	return messages
}

// writeMessagesToFile writes messages to a file
func writeMessagesToFile(messages []SyslogMessage, filename string, format string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	
	writer := bufio.NewWriter(file)
	defer writer.Flush()
	
	for _, msg := range messages {
		var formattedMsg string
		if format == "rfc5424" {
			formattedMsg = msg.FormatRFC5424()
		} else {
			formattedMsg = msg.FormatRFC3164()
		}
		
		_, err := writer.WriteString(formattedMsg + "\n")
		if err != nil {
			return err
		}
	}
	
	return nil
}

// generateStatistics generates statistics for the generated messages
func generateStatistics(messages []SyslogMessage, deviceType string) {
	eventCounts := make(map[string]int)
	severityCounts := make(map[int]int)
	programCounts := make(map[string]int)
	
	for _, msg := range messages {
		// Count by program (approximate event type)
		programCounts[msg.Program]++
		severityCounts[msg.Severity]++
		
		// Categorize by message content
		msgLower := strings.ToLower(msg.Message)
		if strings.Contains(msgLower, "login") || strings.Contains(msgLower, "authentication") || strings.Contains(msgLower, "ssh") {
			eventCounts["Login Events"]++
		} else if strings.Contains(msgLower, "bgp") {
			eventCounts["BGP Events"]++
		} else if strings.Contains(msgLower, "ospf") {
			eventCounts["OSPF Events"]++
		} else if strings.Contains(msgLower, "latency") || strings.Contains(msgLower, "rtt") || strings.Contains(msgLower, "ping") {
			eventCounts["Latency Events"]++
		} else if strings.Contains(msgLower, "attack") || strings.Contains(msgLower, "threat") || strings.Contains(msgLower, "malware") || strings.Contains(msgLower, "security") {
			eventCounts["Security Events"]++
		} else if strings.Contains(msgLower, "interface") || strings.Contains(msgLower, "link") {
			eventCounts["Interface Events"]++
		} else if strings.Contains(msgLower, "cpu") || strings.Contains(msgLower, "memory") || strings.Contains(msgLower, "temperature") || strings.Contains(msgLower, "disk") {
			eventCounts["System Events"]++
		} else if strings.Contains(msgLower, "route") || strings.Contains(msgLower, "routing") {
			eventCounts["Routing Events"]++
		} else if strings.Contains(msgLower, "vpn") || strings.Contains(msgLower, "globalprotect") {
			eventCounts["VPN Events"]++
		} else {
			eventCounts["Other Events"]++
		}
	}
	
	fmt.Printf("\n=== %s Statistics ===\n", deviceType)
	fmt.Printf("Total Messages: %d\n", len(messages))
	
	fmt.Println("\nEvent Type Distribution:")
	for eventType, count := range eventCounts {
		percentage := float64(count) / float64(len(messages)) * 100
		fmt.Printf("  %-20s: %d (%.1f%%)\n", eventType, count, percentage)
	}
	
	fmt.Println("\nSeverity Distribution:")
	severityNames := map[int]string{
		0: "Emergency", 1: "Alert", 2: "Critical", 3: "Error",
		4: "Warning", 5: "Notice", 6: "Info", 7: "Debug",
	}
	for i := 0; i <= 7; i++ {
		if count, exists := severityCounts[i]; exists {
			percentage := float64(count) / float64(len(messages)) * 100
			fmt.Printf("  %-12s (%d): %d (%.1f%%)\n", severityNames[i], i, count, percentage)
		}
	}
	
	fmt.Println("\nTop Programs/Services:")
	type programCount struct {
		program string
		count   int
	}
	var programs []programCount
	for program, count := range programCounts {
		programs = append(programs, programCount{program, count})
	}
	
	// Sort by count (simple bubble sort for small datasets)
	for i := 0; i < len(programs); i++ {
		for j := 0; j < len(programs)-1-i; j++ {
			if programs[j].count < programs[j+1].count {
				programs[j], programs[j+1] = programs[j+1], programs[j]
			}
		}
	}
	
	// Show top 10
	maxShow := 10
	if len(programs) < maxShow {
		maxShow = len(programs)
	}
	for i := 0; i < maxShow; i++ {
		percentage := float64(programs[i].count) / float64(len(messages)) * 100
		fmt.Printf("  %-15s: %d (%.1f%%)\n", programs[i].program, programs[i].count, percentage)
	}
}

// sendToSyslogServer sends messages to a remote syslog server (optional)
func sendToSyslogServer(messages []SyslogMessage, serverAddr string) error {
	conn, err := net.Dial("udp", serverAddr)
	if err != nil {
		return fmt.Errorf("failed to connect to syslog server: %v", err)
	}
	defer conn.Close()
	
	for _, msg := range messages {
		formattedMsg := msg.FormatRFC3164()
		_, err := conn.Write([]byte(formattedMsg))
		if err != nil {
			return fmt.Errorf("failed to send message: %v", err)
		}
		
		// Small delay to avoid overwhelming the server
		time.Sleep(time.Millisecond * 10)
	}
	
	return nil
}

// generateRealTimeMessages generates messages continuously for real-time simulation
func (sg *SyslogGenerator) generateRealTimeMessages(deviceType DeviceType, ratePerSecond int, duration time.Duration, outputChan chan<- SyslogMessage) {
	ticker := time.NewTicker(time.Second / time.Duration(ratePerSecond))
	defer ticker.Stop()
	
	endTime := time.Now().Add(duration)
	deviceNames := sg.deviceNames[deviceType]
	
	for time.Now().Before(endTime) {
		select {
		case <-ticker.C:
			hostname := deviceNames[sg.rand.Intn(len(deviceNames))]
			
			// Random event type
			eventTypes := []EventType{
				LoginEvent, BGPFlapEvent, OSPFFlapEvent, LatencyEvent,
				SecurityEvent, TCPAttackEvent, InterfaceEvent, SystemEvent,
				RoutingEvent, VPNEvent,
			}
			eventType := eventTypes[sg.rand.Intn(len(eventTypes))]
			
			var msg SyslogMessage
			switch deviceType {
			case CiscoIOSXE:
				msg = sg.generateCiscoIOSXEMessage(eventType, hostname)
			case PaloAltoFirewall:
				msg = sg.generatePaloAltoMessage(eventType, hostname)
			case JuniperRouter:
				msg = sg.generateJuniperMessage(eventType, hostname)
			}
			
			// Update timestamp to current time for real-time simulation
			msg.Timestamp = time.Now().Format("Jan 02 15:04:05")
			
			outputChan <- msg
		}
	}
}

// createSampleConfigFile creates a sample configuration file
func createSampleConfigFile() error {
	config := `# Syslog Generator Configuration
# Device counts for message generation
cisco_iosxe_count: 10000
palo_alto_count: 10000
juniper_count: 10000

# Output settings
output_format: rfc3164  # rfc3164 or rfc5424
output_directory: ./syslog_output/
separate_files: true    # Create separate files per device type

# Real-time generation settings
real_time_mode: false
messages_per_second: 10
duration_minutes: 60

# Remote syslog server (optional)
syslog_server: ""       # Example: "192.168.1.100:514"
send_to_server: false

# Event distribution percentages
event_distribution:
  login_events: 15
  bgp_flap_events: 5
  ospf_flap_events: 5
  latency_events: 10
  security_events: 15
  tcp_attack_events: 8
  interface_events: 12
  system_events: 20
  routing_events: 7
  vpn_events: 3
`
	
	return os.WriteFile("syslog_config.yaml", []byte(config), 0644)
}

func main() {
	fmt.Println("=== Network Device Syslog Generator ===")
	fmt.Println("Generating realistic syslog messages for network devices...")
	
	// Create output directory
	if err := os.MkdirAll("syslog_output", 0755); err != nil {
		log.Fatalf("Failed to create output directory: %v", err)
	}
	
	// Create sample configuration file
	if err := createSampleConfigFile(); err != nil {
		log.Printf("Warning: Could not create sample config file: %v", err)
	}
	
	// Initialize generator
	generator := NewSyslogGenerator()
	
	// Generate messages for each device type
	deviceTypes := []struct {
		deviceType DeviceType
		name       string
		filename   string
	}{
		{CiscoIOSXE, "Cisco IOS-XE", "cisco_iosxe_syslog.log"},
		{PaloAltoFirewall, "Palo Alto Firewall", "paloalto_firewall_syslog.log"},
		{JuniperRouter, "Juniper Router", "juniper_router_syslog.log"},
	}
	
	messagesPerDevice := 10000
	format := "rfc3164" // or "rfc5424"
	
	fmt.Printf("Generating %d messages per device type...\n", messagesPerDevice)
	
	var allMessages []SyslogMessage
	
	for _, device := range deviceTypes {
		fmt.Printf("\nGenerating %s messages...", device.name)
		startTime := time.Now()
		
		messages := generator.generateMessages(device.deviceType, messagesPerDevice)
		
		// Write to individual device file
		filename := fmt.Sprintf("syslog_output/%s", device.filename)
		if err := writeMessagesToFile(messages, filename, format); err != nil {
			log.Fatalf("Failed to write %s messages: %v", device.name, err)
		}
		
		duration := time.Since(startTime)
		fmt.Printf(" ✓ Complete! (%v)\n", duration)
		fmt.Printf("   Output: %s (%d messages)\n", filename, len(messages))
		
		// Generate statistics
		generateStatistics(messages, device.name)
		
		// Add to combined collection
		allMessages = append(allMessages, messages...)
	}
	
	// Write combined file with all messages mixed together
	fmt.Printf("\nCreating combined syslog file with all %d messages...", len(allMessages))
	
	// Shuffle all messages to simulate real mixed environment
	for i := range allMessages {
		j := generator.rand.Intn(i + 1)
		allMessages[i], allMessages[j] = allMessages[j], allMessages[i]
	}
	
	if err := writeMessagesToFile(allMessages, "syslog_output/combined_all_devices.log", format); err != nil {
		log.Fatalf("Failed to write combined file: %v", err)
	}
	fmt.Println(" ✓ Complete!")
	
	// Generate overall statistics
	fmt.Println("\n" + strings.Repeat("=", 60))
	generateStatistics(allMessages, "All Devices Combined")
	
	// Demo: Real-time message generation
	fmt.Println("\n=== Real-time Message Generation Demo ===")
	fmt.Println("Generating real-time messages for 30 seconds (5 msg/sec)...")
	
	realTimeChan := make(chan SyslogMessage, 100)
	
	// Start real-time generation for Cisco devices
	go generator.generateRealTimeMessages(CiscoIOSXE, 5, 30*time.Second, realTimeChan)
	
	// Process real-time messages
	go func() {
		realTimeFile, err := os.Create("syslog_output/realtime_demo.log")
		if err != nil {
			log.Printf("Failed to create real-time file: %v", err)
			return
		}
		defer realTimeFile.Close()
		
		writer := bufio.NewWriter(realTimeFile)
		defer writer.Flush()
		
		messageCount := 0
		for msg := range realTimeChan {
			formattedMsg := msg.FormatRFC3164()
			fmt.Printf("[REALTIME] %s\n", formattedMsg)
			writer.WriteString(formattedMsg + "\n")
			messageCount++
		}
		
		fmt.Printf("\nReal-time demo complete: %d messages generated\n", messageCount)
	}()
	
	// Wait for real-time demo to complete
	time.Sleep(35 * time.Second)
	close(realTimeChan)
	time.Sleep(1 * time.Second) // Allow file writing to complete
	
	// Optional: Send to remote syslog server
	fmt.Println("\n=== Remote Syslog Server Demo ===")
	syslogServer := "" // Example: "192.168.1.100:514"
	
	if syslogServer != "" {
		fmt.Printf("Sending sample messages to %s...\n", syslogServer)
		sampleMessages := generator.generateMessages(CiscoIOSXE, 10)
		if err := sendToSyslogServer(sampleMessages, syslogServer); err != nil {
			log.Printf("Failed to send to syslog server: %v", err)
		} else {
			fmt.Println("✓ Messages sent successfully!")
		}
	} else {
		fmt.Println("No syslog server configured. Set syslogServer variable to test.")
	}
	
	// Summary
	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("GENERATION COMPLETE!")
	fmt.Println(strings.Repeat("=", 60))
	fmt.Printf("Total messages generated: %d\n", len(allMessages))
	fmt.Printf("Output directory: ./syslog_output/\n")
	fmt.Printf("Format used: %s\n", strings.ToUpper(format))
	
	fmt.Println("\nGenerated files:")
	fmt.Println("  - cisco_iosxe_syslog.log      (Cisco IOS-XE messages)")
	fmt.Println("  - paloalto_firewall_syslog.log (Palo Alto messages)")
	fmt.Println("  - juniper_router_syslog.log    (Juniper messages)")
	fmt.Println("  - combined_all_devices.log     (All messages mixed)")
	fmt.Println("  - realtime_demo.log            (Real-time demo messages)")
	fmt.Println("  - syslog_config.yaml           (Sample configuration)")
	
	fmt.Println("\nMessage types included:")
	fmt.Println("  ✓ Login/Authentication events")
	fmt.Println("  ✓ BGP neighbor flaps")
	fmt.Println("  ✓ OSPF neighbor changes")
	fmt.Println("  ✓ Network latency alerts")
	fmt.Println("  ✓ Security threats and attacks")
	fmt.Println("  ✓ TCP attack detection")
	fmt.Println("  ✓ Interface up/down events")
	fmt.Println("  ✓ System resource monitoring")
	fmt.Println("  ✓ Routing table changes")
	fmt.Println("  ✓ VPN connection events")
	
	fmt.Println("\nUse these logs for:")
	fmt.Println("  • Testing log analysis tools")
	fmt.Println("  • Training machine learning models")
	fmt.Println("  • Developing security detection rules")
	fmt.Println("  • Simulating network operations centers")
	fmt.Println("  • Performance testing log processing systems")
}