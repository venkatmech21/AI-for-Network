package main

import (
	"bufio"
	"crypto/md5"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

// NormalizedSyslogMessage represents a cleaned and normalized syslog message
type NormalizedSyslogMessage struct {
	ID                string    `json:"id"`
	OriginalMessage   string    `json:"original_message"`
	Timestamp         time.Time `json:"timestamp"`
	NormalizedTime    string    `json:"normalized_time"`
	Facility          int       `json:"facility"`
	Severity          int       `json:"severity"`
	SeverityName      string    `json:"severity_name"`
	Priority          int       `json:"priority"`
	Hostname          string    `json:"hostname"`
	Program           string    `json:"program"`
	PID               int       `json:"pid"`
	Message           string    `json:"message"`
	DeviceType        string    `json:"device_type"`
	EventCategory     string    `json:"event_category"`
	EventSubcategory  string    `json:"event_subcategory"`
	SourceIP          string    `json:"source_ip,omitempty"`
	DestinationIP     string    `json:"destination_ip,omitempty"`
	SourcePort        int       `json:"source_port,omitempty"`
	DestinationPort   int       `json:"destination_port,omitempty"`
	Protocol          string    `json:"protocol,omitempty"`
	Username          string    `json:"username,omitempty"`
	Interface         string    `json:"interface,omitempty"`
	ThreatName        string    `json:"threat_name,omitempty"`
	BGPPeer           string    `json:"bgp_peer,omitempty"`
	ASNumber          int       `json:"as_number,omitempty"`
	OSPFArea          string    `json:"ospf_area,omitempty"`
	RiskScore         int       `json:"risk_score"`
	IsSecurity        bool      `json:"is_security"`
	IsAuthentication  bool      `json:"is_authentication"`
	IsNetworkEvent    bool      `json:"is_network_event"`
	IsSystemEvent     bool      `json:"is_system_event"`
	ProcessedAt       time.Time `json:"processed_at"`
	DataQualityIssues []string  `json:"data_quality_issues,omitempty"`
}

// CleaningStats tracks data cleaning statistics
type CleaningStats struct {
	TotalProcessed       int               `json:"total_processed"`
	ValidMessages        int               `json:"valid_messages"`
	InvalidMessages      int               `json:"invalid_messages"`
	DuplicatesRemoved    int               `json:"duplicates_removed"`
	MalformedTimestamps  int               `json:"malformed_timestamps"`
	InvalidPriorities    int               `json:"invalid_priorities"`
	MissingHostnames     int               `json:"missing_hostnames"`
	NormalizedIPs        int               `json:"normalized_ips"`
	ExtractedFields      map[string]int    `json:"extracted_fields"`
	DeviceTypeStats      map[string]int    `json:"device_type_stats"`
	EventCategoryStats   map[string]int    `json:"event_category_stats"`
	SeverityStats        map[string]int    `json:"severity_stats"`
	ProcessingStartTime  time.Time         `json:"processing_start_time"`
	ProcessingEndTime    time.Time         `json:"processing_end_time"`
	ProcessingDuration   time.Duration     `json:"processing_duration"`
}

// SyslogCleaner handles the cleaning and normalization process
type SyslogCleaner struct {
	severityNames     map[int]string
	devicePatterns    map[string]*regexp.Regexp
	ipPattern         *regexp.Regexp
	portPattern       *regexp.Regexp
	authPatterns      []*regexp.Regexp
	securityPatterns  []*regexp.Regexp
	networkPatterns   []*regexp.Regexp
	systemPatterns    []*regexp.Regexp
	threatPatterns    []*regexp.Regexp
	bgpPatterns       []*regexp.Regexp
	ospfPatterns      []*regexp.Regexp
	interfacePatterns []*regexp.Regexp
	usernamePatterns  []*regexp.Regexp
	duplicateHashes   map[string]bool
	stats             *CleaningStats
}

// NewSyslogCleaner creates a new syslog cleaner with predefined patterns
func NewSyslogCleaner() *SyslogCleaner {
	cleaner := &SyslogCleaner{
		severityNames: map[int]string{
			0: "Emergency", 1: "Alert", 2: "Critical", 3: "Error",
			4: "Warning", 5: "Notice", 6: "Info", 7: "Debug",
		},
		devicePatterns: map[string]*regexp.Regexp{
			"cisco":     regexp.MustCompile(`(?i)(RTR|SW|CORE|EDGE|DIST|ACCESS|WAN|MPLS)`),
			"paloalto":  regexp.MustCompile(`(?i)(PA-FW|PA-|paloalto)`),
			"juniper":   regexp.MustCompile(`(?i)(JUN-|MX-|EX-|SRX-|QFX-)`),
			"fortigate": regexp.MustCompile(`(?i)(FGT-|FortiGate)`),
			"checkpoint": regexp.MustCompile(`(?i)(CP-|checkpoint)`),
		},
		duplicateHashes: make(map[string]bool),
		stats: &CleaningStats{
			ExtractedFields:    make(map[string]int),
			DeviceTypeStats:    make(map[string]int),
			EventCategoryStats: make(map[string]int),
			SeverityStats:      make(map[string]int),
			ProcessingStartTime: time.Now(),
		},
	}
	
	// Compile regex patterns
	cleaner.compilePatterns()
	
	return cleaner
}

// compilePatterns compiles all regex patterns used for data extraction
func (sc *SyslogCleaner) compilePatterns() {
	// IP address pattern (IPv4 and IPv6)
	sc.ipPattern = regexp.MustCompile(`(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)|(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}`)
	
	// Port pattern
	sc.portPattern = regexp.MustCompile(`(?:port|:)\s*(\d{1,5})`)
	
	// Authentication patterns
	sc.authPatterns = []*regexp.Regexp{
		regexp.MustCompile(`(?i)login|authentication|ssh|telnet|console`),
		regexp.MustCompile(`(?i)user\s+(?:'([^']+)'|(\S+))`),
		regexp.MustCompile(`(?i)(?:succeeded|failed|accepted|denied)`),
		regexp.MustCompile(`(?i)password|certificate|key`),
	}
	
	// Security threat patterns
	sc.securityPatterns = []*regexp.Regexp{
		regexp.MustCompile(`(?i)attack|threat|malware|virus|trojan|backdoor`),
		regexp.MustCompile(`(?i)intrusion|breach|compromise|exploit`),
		regexp.MustCompile(`(?i)scan|probe|reconnaissance|brute.?force`),
		regexp.MustCompile(`(?i)ddos|dos|flood|amplification`),
		regexp.MustCompile(`(?i)botnet|c2|command.?control|exfiltration`),
	}
	
	// Network event patterns
	sc.networkPatterns = []*regexp.Regexp{
		regexp.MustCompile(`(?i)bgp|ospf|eigrp|isis|rip`),
		regexp.MustCompile(`(?i)interface|link|port|ethernet`),
		regexp.MustCompile(`(?i)route|routing|nexthop|gateway`),
		regexp.MustCompile(`(?i)vlan|trunk|spanning.?tree|stp`),
		regexp.MustCompile(`(?i)up|down|flap|change|transition`),
	}
	
	// System event patterns
	sc.systemPatterns = []*regexp.Regexp{
		regexp.MustCompile(`(?i)cpu|memory|disk|temperature|fan`),
		regexp.MustCompile(`(?i)utilization|usage|threshold|alarm`),
		regexp.MustCompile(`(?i)power|supply|redundancy|backup`),
		regexp.MustCompile(`(?i)config|configuration|startup|shutdown`),
	}
	
	// Threat name extraction patterns
	sc.threatPatterns = []*regexp.Regexp{
		regexp.MustCompile(`(?i)threat[^:]*:\s*([^,\s]+)`),
		regexp.MustCompile(`(?i)malware[^:]*:\s*([^,\s]+)`),
		regexp.MustCompile(`(?i)signature[^:]*:\s*([^,\s]+)`),
		regexp.MustCompile(`(?i)"([^"]*(?:malware|threat|exploit|trojan)[^"]*)"`),
	}
	
	// BGP patterns
	sc.bgpPatterns = []*regexp.Regexp{
		regexp.MustCompile(`(?i)bgp.*neighbor\s+([0-9.]+)`),
		regexp.MustCompile(`(?i)AS\s*(\d+)`),
		regexp.MustCompile(`(?i)peer\s+([0-9.]+)`),
	}
	
	// OSPF patterns
	sc.ospfPatterns = []*regexp.Regexp{
		regexp.MustCompile(`(?i)ospf.*neighbor\s+([0-9.]+)`),
		regexp.MustCompile(`(?i)area\s+([0-9.]+)`),
	}
	
	// Interface patterns
	sc.interfacePatterns = []*regexp.Regexp{
		regexp.MustCompile(`(?i)interface\s+([a-zA-Z0-9\/\.-]+)`),
		regexp.MustCompile(`(?i)on\s+([a-zA-Z0-9\/\.-]+)`),
		regexp.MustCompile(`(?i)(GigabitEthernet|TenGigabitEthernet|FastEthernet|Serial|Loopback|Tunnel|Vlan|ethernet|ge-|xe-|et-)\S*`),
	}
	
	// Username extraction patterns
	sc.usernamePatterns = []*regexp.Regexp{
		regexp.MustCompile(`(?i)user\s+(?:'([^']+)'|(\S+))`),
		regexp.MustCompile(`(?i)for\s+user\s+(?:'([^']+)'|(\S+))`),
		regexp.MustCompile(`(?i)User:\s+(\S+)`),
	}
}

// parseSyslogMessage parses a raw syslog line into its components
func (sc *SyslogCleaner) parseSyslogMessage(line string) (*NormalizedSyslogMessage, error) {
	line = strings.TrimSpace(line)
	if line == "" {
		return nil, fmt.Errorf("empty line")
	}
	
	msg := &NormalizedSyslogMessage{
		OriginalMessage:   line,
		ProcessedAt:       time.Now(),
		DataQualityIssues: []string{},
	}
	
	// Generate unique ID
	hash := md5.Sum([]byte(line + fmt.Sprintf("%d", time.Now().UnixNano())))
	msg.ID = fmt.Sprintf("%x", hash)[:16]
	
	// Parse RFC 3164 format: <priority>timestamp hostname program[pid]: message
	rfc3164Pattern := regexp.MustCompile(`^<(\d+)>(\w+\s+\d+\s+\d+:\d+:\d+)\s+(\S+)\s+([^:\[]+)(?:\[(\d+)\])?:\s*(.*)$`)
	
	// Parse RFC 5424 format: <priority>version timestamp hostname program pid msgid structured-data message
	rfc5424Pattern := regexp.MustCompile(`^<(\d+)>(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(.*)$`)
	
	var matches []string
	isRFC5424 := false
	
	if matches = rfc5424Pattern.FindStringSubmatch(line); matches != nil {
		isRFC5424 = true
	} else if matches = rfc3164Pattern.FindStringSubmatch(line); matches == nil {
		// Try fallback pattern for malformed messages
		fallbackPattern := regexp.MustCompile(`^<(\d+)>(.*)$`)
		if fallbackMatches := fallbackPattern.FindStringSubmatch(line); fallbackMatches != nil {
			matches = []string{line, fallbackMatches[1], "", "unknown", "unknown", "", fallbackMatches[2]}
			msg.DataQualityIssues = append(msg.DataQualityIssues, "malformed_syslog_format")
		} else {
			return nil, fmt.Errorf("unable to parse syslog message format")
		}
	}
	
	// Parse priority
	if priority, err := strconv.Atoi(matches[1]); err == nil {
		if priority < 0 || priority > 191 {
			msg.DataQualityIssues = append(msg.DataQualityIssues, "invalid_priority_range")
			priority = 134 // Default to local0.info
		}
		msg.Priority = priority
		msg.Facility = priority / 8
		msg.Severity = priority % 8
		msg.SeverityName = sc.severityNames[msg.Severity]
	} else {
		msg.DataQualityIssues = append(msg.DataQualityIssues, "unparseable_priority")
		sc.stats.InvalidPriorities++
	}
	
	// Parse timestamp
	if isRFC5424 {
		// RFC 5424 format
		if timestamp, err := time.Parse("2006-01-02T15:04:05.000Z", matches[3]); err == nil {
			msg.Timestamp = timestamp.UTC()
		} else if timestamp, err := time.Parse("2006-01-02T15:04:05Z", matches[3]); err == nil {
			msg.Timestamp = timestamp.UTC()
		} else {
			msg.DataQualityIssues = append(msg.DataQualityIssues, "unparseable_timestamp")
			sc.stats.MalformedTimestamps++
		}
		msg.Hostname = matches[4]
		msg.Program = matches[5]
		if pid, err := strconv.Atoi(matches[6]); err == nil {
			msg.PID = pid
		}
		msg.Message = matches[8]
	} else {
		// RFC 3164 format
		if timestamp, err := sc.parseRFC3164Timestamp(matches[2]); err == nil {
			msg.Timestamp = timestamp
		} else {
			msg.DataQualityIssues = append(msg.DataQualityIssues, "unparseable_timestamp")
			sc.stats.MalformedTimestamps++
		}
		msg.Hostname = matches[3]
		msg.Program = matches[4]
		if len(matches) > 5 && matches[5] != "" {
			if pid, err := strconv.Atoi(matches[5]); err == nil {
				msg.PID = pid
			}
		}
		if len(matches) > 6 {
			msg.Message = matches[6]
		}
	}
	
	// Normalize timestamp format
	if !msg.Timestamp.IsZero() {
		msg.NormalizedTime = msg.Timestamp.UTC().Format("2006-01-02T15:04:05.000Z")
	}
	
	// Validate and clean hostname
	if msg.Hostname == "" || msg.Hostname == "-" {
		msg.DataQualityIssues = append(msg.DataQualityIssues, "missing_hostname")
		sc.stats.MissingHostnames++
	} else {
		msg.Hostname = sc.normalizeHostname(msg.Hostname)
	}
	
	return msg, nil
}

// parseRFC3164Timestamp parses RFC 3164 timestamp format
func (sc *SyslogCleaner) parseRFC3164Timestamp(timeStr string) (time.Time, error) {
	// Try different RFC 3164 timestamp formats
	formats := []string{
		"Jan 02 15:04:05",
		"Jan  2 15:04:05",
		"2006 Jan 02 15:04:05",
		"2006 Jan  2 15:04:05",
	}
	
	currentYear := time.Now().Year()
	
	for _, format := range formats {
		if t, err := time.Parse(format, timeStr); err == nil {
			// RFC 3164 doesn't include year, so add current year
			if !strings.Contains(format, "2006") {
				t = t.AddDate(currentYear-t.Year(), 0, 0)
			}
			return t.UTC(), nil
		}
	}
	
	return time.Time{}, fmt.Errorf("unable to parse timestamp: %s", timeStr)
}

// normalizeHostname cleans and standardizes hostname
func (sc *SyslogCleaner) normalizeHostname(hostname string) string {
	// Remove domain suffixes
	if parts := strings.Split(hostname, "."); len(parts) > 1 {
		hostname = parts[0]
	}
	
	// Standardize case
	hostname = strings.ToUpper(hostname)
	
	// Remove invalid characters
	invalidChars := regexp.MustCompile(`[^A-Z0-9\-_]`)
	hostname = invalidChars.ReplaceAllString(hostname, "")
	
	return hostname
}

// classifyDevice determines the device type based on hostname and message content
func (sc *SyslogCleaner) classifyDevice(msg *NormalizedSyslogMessage) {
	hostname := strings.ToLower(msg.Hostname)
	message := strings.ToLower(msg.Message)
	program := strings.ToLower(msg.Program)
	
	// Check hostname patterns
	for deviceType, pattern := range sc.devicePatterns {
		if pattern.MatchString(hostname) {
			msg.DeviceType = deviceType
			sc.stats.DeviceTypeStats[deviceType]++
			return
		}
	}
	
	// Check program/service indicators
	if strings.Contains(program, "bgp") || strings.Contains(program, "ospf") || 
	   strings.Contains(program, "rpd") || strings.Contains(program, "routing") {
		msg.DeviceType = "router"
	} else if strings.Contains(program, "firewall") || strings.Contains(program, "threat") ||
			  strings.Contains(program, "security") || strings.Contains(program, "fw") {
		msg.DeviceType = "firewall"
	} else if strings.Contains(program, "switch") || strings.Contains(program, "vlan") ||
			  strings.Contains(program, "spanning") {
		msg.DeviceType = "switch"
	} else if strings.Contains(message, "globalprotect") || strings.Contains(message, "vpn") {
		msg.DeviceType = "firewall"
	} else {
		msg.DeviceType = "unknown"
	}
	
	sc.stats.DeviceTypeStats[msg.DeviceType]++
}

// categorizeEvent determines the event category and subcategory
func (sc *SyslogCleaner) categorizeEvent(msg *NormalizedSyslogMessage) {
	message := strings.ToLower(msg.Message)
	program := strings.ToLower(msg.Program)
	
	// Authentication events
	if sc.matchesAnyPattern(sc.authPatterns, message) || sc.matchesAnyPattern(sc.authPatterns, program) {
		msg.EventCategory = "authentication"
		msg.IsAuthentication = true
		
		if strings.Contains(message, "success") || strings.Contains(message, "accept") {
			msg.EventSubcategory = "login_success"
		} else if strings.Contains(message, "fail") || strings.Contains(message, "denied") {
			msg.EventSubcategory = "login_failure"
		} else {
			msg.EventSubcategory = "authentication_event"
		}
	} else if sc.matchesAnyPattern(sc.securityPatterns, message) {
		// Security events
		msg.EventCategory = "security"
		msg.IsSecurity = true
		
		if strings.Contains(message, "attack") || strings.Contains(message, "threat") {
			msg.EventSubcategory = "threat_detection"
		} else if strings.Contains(message, "scan") || strings.Contains(message, "probe") {
			msg.EventSubcategory = "reconnaissance"
		} else if strings.Contains(message, "malware") || strings.Contains(message, "virus") {
			msg.EventSubcategory = "malware_detection"
		} else {
			msg.EventSubcategory = "security_event"
		}
	} else if sc.matchesAnyPattern(sc.networkPatterns, message) {
		// Network events
		msg.EventCategory = "network"
		msg.IsNetworkEvent = true
		
		if strings.Contains(message, "bgp") {
			msg.EventSubcategory = "bgp_event"
		} else if strings.Contains(message, "ospf") {
			msg.EventSubcategory = "ospf_event"
		} else if strings.Contains(message, "interface") || strings.Contains(message, "link") {
			msg.EventSubcategory = "interface_event"
		} else if strings.Contains(message, "route") {
			msg.EventSubcategory = "routing_event"
		} else {
			msg.EventSubcategory = "network_event"
		}
	} else if sc.matchesAnyPattern(sc.systemPatterns, message) {
		// System events
		msg.EventCategory = "system"
		msg.IsSystemEvent = true
		
		if strings.Contains(message, "cpu") || strings.Contains(message, "memory") {
			msg.EventSubcategory = "resource_monitoring"
		} else if strings.Contains(message, "temperature") || strings.Contains(message, "fan") {
			msg.EventSubcategory = "hardware_monitoring"
		} else if strings.Contains(message, "config") {
			msg.EventSubcategory = "configuration_event"
		} else {
			msg.EventSubcategory = "system_event"
		}
	} else {
		// Other events
		msg.EventCategory = "other"
		msg.EventSubcategory = "unclassified"
	}
	
	sc.stats.EventCategoryStats[msg.EventCategory]++
}

// extractFields extracts specific fields from the message content
func (sc *SyslogCleaner) extractFields(msg *NormalizedSyslogMessage) {
	message := msg.Message
	
	// Extract IP addresses
	ips := sc.ipPattern.FindAllString(message, -1)
	if len(ips) > 0 {
		// Deduplicate and validate IPs
		uniqueIPs := make(map[string]bool)
		for _, ip := range ips {
			if net.ParseIP(ip) != nil {
				uniqueIPs[ip] = true
			}
		}
		
		// Assign first two unique IPs as source and destination
		var validIPs []string
		for ip := range uniqueIPs {
			validIPs = append(validIPs, ip)
		}
		sort.Strings(validIPs) // Consistent ordering
		
		if len(validIPs) > 0 {
			msg.SourceIP = validIPs[0]
			sc.stats.ExtractedFields["source_ip"]++
			sc.stats.NormalizedIPs++
		}
		if len(validIPs) > 1 {
			msg.DestinationIP = validIPs[1]
			sc.stats.ExtractedFields["destination_ip"]++
		}
	}
	
	// Extract ports
	ports := sc.portPattern.FindAllStringSubmatch(message, -1)
	if len(ports) > 0 {
		if port, err := strconv.Atoi(ports[0][1]); err == nil && port > 0 && port <= 65535 {
			msg.DestinationPort = port
			sc.stats.ExtractedFields["destination_port"]++
		}
	}
	
	// Extract protocol
	protocolPattern := regexp.MustCompile(`(?i)\b(tcp|udp|icmp|esp|ah|gre|ospf|bgp|eigrp)\b`)
	if match := protocolPattern.FindString(message); match != "" {
		msg.Protocol = strings.ToUpper(match)
		sc.stats.ExtractedFields["protocol"]++
	}
	
	// Extract username
	for _, pattern := range sc.usernamePatterns {
		if matches := pattern.FindStringSubmatch(message); matches != nil {
			for i := 1; i < len(matches); i++ {
				if matches[i] != "" {
					msg.Username = matches[i]
					sc.stats.ExtractedFields["username"]++
					break
				}
			}
			if msg.Username != "" {
				break
			}
		}
	}
	
	// Extract interface
	for _, pattern := range sc.interfacePatterns {
		if matches := pattern.FindStringSubmatch(message); matches != nil {
			msg.Interface = matches[1]
			sc.stats.ExtractedFields["interface"]++
			break
		}
	}
	
	// Extract threat name
	for _, pattern := range sc.threatPatterns {
		if matches := pattern.FindStringSubmatch(message); matches != nil {
			msg.ThreatName = strings.Trim(matches[1], `"'`)
			sc.stats.ExtractedFields["threat_name"]++
			break
		}
	}
	
	// Extract BGP peer and AS number
	for _, pattern := range sc.bgpPatterns {
		if matches := pattern.FindStringSubmatch(message); matches != nil {
			if net.ParseIP(matches[1]) != nil {
				msg.BGPPeer = matches[1]
				sc.stats.ExtractedFields["bgp_peer"]++
			}
		}
	}
	
	asPattern := regexp.MustCompile(`(?i)AS\s*(\d+)`)
	if matches := asPattern.FindStringSubmatch(message); matches != nil {
		if asNum, err := strconv.Atoi(matches[1]); err == nil {
			msg.ASNumber = asNum
			sc.stats.ExtractedFields["as_number"]++
		}
	}
	
	// Extract OSPF area
	for _, pattern := range sc.ospfPatterns {
		if matches := pattern.FindStringSubmatch(message); matches != nil {
			msg.OSPFArea = matches[1]
			sc.stats.ExtractedFields["ospf_area"]++
			break
		}
	}
}

// calculateRiskScore assigns a risk score based on event characteristics
func (sc *SyslogCleaner) calculateRiskScore(msg *NormalizedSyslogMessage) {
	score := 0
	
	// Base score by severity
	switch msg.Severity {
	case 0, 1: // Emergency, Alert
		score += 50
	case 2: // Critical
		score += 40
	case 3: // Error
		score += 30
	case 4: // Warning
		score += 20
	case 5: // Notice
		score += 10
	case 6, 7: // Info, Debug
		score += 5
	}
	
	// Additional score for security events
	if msg.IsSecurity {
		score += 30
		
		// Higher score for specific threats
		if msg.ThreatName != "" {
			score += 20
		}
		
		// Higher score for authentication failures
		if msg.IsAuthentication && strings.Contains(strings.ToLower(msg.Message), "fail") {
			score += 15
		}
	}
	
	// Score for suspicious ports
	suspiciousPorts := map[int]bool{
		22: true, 23: true, 1433: true, 3389: true, 5900: true,
	}
	if suspiciousPorts[msg.DestinationPort] {
		score += 10
	}
	
	// Score for external IP addresses (simplified check)
	if msg.SourceIP != "" && !strings.HasPrefix(msg.SourceIP, "192.168.") && 
	   !strings.HasPrefix(msg.SourceIP, "10.") && !strings.HasPrefix(msg.SourceIP, "172.16.") {
		score += 15
	}
	
	// Cap the score at 100
	if score > 100 {
		score = 100
	}
	
	msg.RiskScore = score
}

// matchesAnyPattern checks if text matches any of the provided patterns
func (sc *SyslogCleaner) matchesAnyPattern(patterns []*regexp.Regexp, text string) bool {
	for _, pattern := range patterns {
		if pattern.MatchString(text) {
			return true
		}
	}
	return false
}

// isDuplicate checks if a message is a duplicate based on content hash
func (sc *SyslogCleaner) isDuplicate(msg *NormalizedSyslogMessage) bool {
	// Create content hash (excluding timestamp and ID)
	content := fmt.Sprintf("%s-%s-%s-%s", msg.Hostname, msg.Program, msg.Message, msg.SeverityName)
	hash := fmt.Sprintf("%x", md5.Sum([]byte(content)))
	
	if sc.duplicateHashes[hash] {
		return true
	}
	
	sc.duplicateHashes[hash] = true
	return false
}

// processFile processes a single syslog file
func (sc *SyslogCleaner) processFile(inputPath string) ([]*NormalizedSyslogMessage, error) {
	file, err := os.Open(inputPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file %s: %v", inputPath, err)
	}
	defer file.Close()
	
	var messages []*NormalizedSyslogMessage
	scanner := bufio.NewScanner(file)
	lineNumber := 0
	
	log.Printf("Processing file: %s", inputPath)
	
	for scanner.Scan() {
		lineNumber++
		sc.stats.TotalProcessed++
		
		line := scanner.Text()
		if line == "" {
			continue
		}
		
		// Parse the syslog message
		msg, err := sc.parseSyslogMessage(line)
		if err != nil {
			sc.stats.InvalidMessages++
			log.Printf("Error parsing line %d: %v", lineNumber, err)
			continue
		}
		
		// Check for duplicates
		if sc.isDuplicate(msg) {
			sc.stats.DuplicatesRemoved++
			continue
		}
		
		// Classify device type
		sc.classifyDevice(msg)
		
		// Categorize event
		sc.categorizeEvent(msg)
		
		// Extract specific fields
		sc.extractFields(msg)
		
		// Calculate risk score
		sc.calculateRiskScore(msg)
		
		// Update severity statistics
		sc.stats.SeverityStats[msg.SeverityName]++
		
		messages = append(messages, msg)
		sc.stats.ValidMessages++
		
		// Progress indicator for large files
		if lineNumber%1000 == 0 {
			log.Printf("Processed %d lines from %s", lineNumber, filepath.Base(inputPath))
		}
	}
	
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading file %s: %v", inputPath, err)
	}
	
	log.Printf("Completed processing %s: %d messages processed, %d valid", 
		filepath.Base(inputPath), lineNumber, len(messages))
	
	return messages, nil
}

// saveAsJSON saves normalized messages as JSON
func (sc *SyslogCleaner) saveAsJSON(messages []*NormalizedSyslogMessage, outputPath string) error {
	file, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("failed to create JSON file %s: %v", outputPath, err)
	}
	defer file.Close()
	
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	
	return encoder.Encode(messages)
}

// saveAsCSV saves normalized messages as CSV
func (sc *SyslogCleaner) saveAsCSV(messages []*NormalizedSyslogMessage, outputPath string) error {
	file, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("failed to create CSV file %s: %v", outputPath, err)
	}
	defer file.Close()
	
	writer := csv.NewWriter(file)
	defer writer.Flush()
	
	// Write header
	header := []string{
		"id", "timestamp", "normalized_time", "facility", "severity", "severity_name",
		"priority", "hostname", "program", "pid", "message", "device_type",
		"event_category", "event_subcategory", "source_ip", "destination_ip",
		"source_port", "destination_port", "protocol", "username", "interface",
		"threat_name", "bgp_peer", "as_number", "ospf_area", "risk_score",
		"is_security", "is_authentication", "is_network_event", "is_system_event",
		"data_quality_issues",
	}
	
	if err := writer.Write(header); err != nil {
		return fmt.Errorf("failed to write CSV header: %v", err)
	}
	
	// Write data rows
	for _, msg := range messages {
		row := []string{
			msg.ID,
			msg.Timestamp.Format("2006-01-02 15:04:05"),
			msg.NormalizedTime,
			strconv.Itoa(msg.Facility),
			strconv.Itoa(msg.Severity),
			msg.SeverityName,
			strconv.Itoa(msg.Priority),
			msg.Hostname,
			msg.Program,
			strconv.Itoa(msg.PID),
			msg.Message,
			msg.DeviceType,
			msg.EventCategory,
			msg.EventSubcategory,
			msg.SourceIP,
			msg.DestinationIP,
			strconv.Itoa(msg.SourcePort),
			strconv.Itoa(msg.DestinationPort),
			msg.Protocol,
			msg.Username,
			msg.Interface,
			msg.ThreatName,
			msg.BGPPeer,
			strconv.Itoa(msg.ASNumber),
			msg.OSPFArea,
			strconv.Itoa(msg.RiskScore),
			strconv.FormatBool(msg.IsSecurity),
			strconv.FormatBool(msg.IsAuthentication),
			strconv.FormatBool(msg.IsNetworkEvent),
			strconv.FormatBool(msg.IsSystemEvent),
			strings.Join(msg.DataQualityIssues, ";"),
		}
		
		if err := writer.Write(row); err != nil {
			return fmt.Errorf("failed to write CSV row: %v", err)
		}
	}
	
	return nil
}

// saveAsNormalizedSyslog saves messages in cleaned syslog format
func (sc *SyslogCleaner) saveAsNormalizedSyslog(messages []*NormalizedSyslogMessage, outputPath string) error {
	file, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("failed to create normalized syslog file %s: %v", outputPath, err)
	}
	defer file.Close()
	
	writer := bufio.NewWriter(file)
	defer writer.Flush()
	
	for _, msg := range messages {
		// Format as RFC 5424 with enhanced fields
		normalizedMsg := fmt.Sprintf("<%d>1 %s %s %s %d - [meta device_type=\"%s\" event_category=\"%s\" risk_score=\"%d\"] %s\n",
			msg.Priority,
			msg.NormalizedTime,
			msg.Hostname,
			msg.Program,
			msg.PID,
			msg.DeviceType,
			msg.EventCategory,
			msg.RiskScore,
			msg.Message,
		)
		
		if _, err := writer.WriteString(normalizedMsg); err != nil {
			return fmt.Errorf("failed to write normalized message: %v", err)
		}
	}
	
	return nil
}

// generateReport creates a comprehensive cleaning report
func (sc *SyslogCleaner) generateReport(outputDir string) error {
	sc.stats.ProcessingEndTime = time.Now()
	sc.stats.ProcessingDuration = sc.stats.ProcessingEndTime.Sub(sc.stats.ProcessingStartTime)
	
	reportPath := filepath.Join(outputDir, "cleaning_report.json")
	file, err := os.Create(reportPath)
	if err != nil {
		return fmt.Errorf("failed to create report file: %v", err)
	}
	defer file.Close()
	
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	
	if err := encoder.Encode(sc.stats); err != nil {
		return fmt.Errorf("failed to write report: %v", err)
	}
	
	// Also create a human-readable summary
	summaryPath := filepath.Join(outputDir, "cleaning_summary.txt")
	summaryFile, err := os.Create(summaryPath)
	if err != nil {
		return fmt.Errorf("failed to create summary file: %v", err)
	}
	defer summaryFile.Close()
	
	summary := fmt.Sprintf(`Syslog Data Cleaning Report
=============================

Processing Summary:
- Total messages processed: %d
- Valid messages: %d
- Invalid messages: %d
- Duplicates removed: %d
- Processing time: %v

Data Quality Issues:
- Malformed timestamps: %d
- Invalid priorities: %d
- Missing hostnames: %d
- Normalized IP addresses: %d

Device Type Distribution:
`, sc.stats.TotalProcessed, sc.stats.ValidMessages, sc.stats.InvalidMessages,
		sc.stats.DuplicatesRemoved, sc.stats.ProcessingDuration,
		sc.stats.MalformedTimestamps, sc.stats.InvalidPriorities,
		sc.stats.MissingHostnames, sc.stats.NormalizedIPs)
	
	for deviceType, count := range sc.stats.DeviceTypeStats {
		percentage := float64(count) / float64(sc.stats.ValidMessages) * 100
		summary += fmt.Sprintf("- %s: %d (%.1f%%)\n", deviceType, count, percentage)
	}
	
	summary += "\nEvent Category Distribution:\n"
	for category, count := range sc.stats.EventCategoryStats {
		percentage := float64(count) / float64(sc.stats.ValidMessages) * 100
		summary += fmt.Sprintf("- %s: %d (%.1f%%)\n", category, count, percentage)
	}
	
	summary += "\nSeverity Distribution:\n"
	for severity, count := range sc.stats.SeverityStats {
		percentage := float64(count) / float64(sc.stats.ValidMessages) * 100
		summary += fmt.Sprintf("- %s: %d (%.1f%%)\n", severity, count, percentage)
	}
	
	summary += "\nExtracted Fields:\n"
	for field, count := range sc.stats.ExtractedFields {
		percentage := float64(count) / float64(sc.stats.ValidMessages) * 100
		summary += fmt.Sprintf("- %s: %d (%.1f%%)\n", field, count, percentage)
	}
	
	if _, err := summaryFile.WriteString(summary); err != nil {
		return fmt.Errorf("failed to write summary: %v", err)
	}
	
	log.Printf("Cleaning report saved to: %s", reportPath)
	log.Printf("Cleaning summary saved to: %s", summaryPath)
	
	return nil
}

// FilterCriteria defines filtering options for processed messages
type FilterCriteria struct {
	MinRiskScore     int
	MaxRiskScore     int
	DeviceTypes      []string
	EventCategories  []string
	SeverityLevels   []int
	SecurityOnly     bool
	AuthOnly         bool
	TimeRange        struct {
		Start time.Time
		End   time.Time
	}
	HasThreatName    bool
	HasUsername      bool
	SourceIPPattern  string
	HostnamePattern  string
}

// filterMessages applies filtering criteria to the normalized messages
func (sc *SyslogCleaner) filterMessages(messages []*NormalizedSyslogMessage, criteria FilterCriteria) []*NormalizedSyslogMessage {
	var filtered []*NormalizedSyslogMessage
	
	for _, msg := range messages {
		// Risk score filter
		if criteria.MinRiskScore > 0 && msg.RiskScore < criteria.MinRiskScore {
			continue
		}
		if criteria.MaxRiskScore > 0 && msg.RiskScore > criteria.MaxRiskScore {
			continue
		}
		
		// Device type filter
		if len(criteria.DeviceTypes) > 0 {
			found := false
			for _, deviceType := range criteria.DeviceTypes {
				if msg.DeviceType == deviceType {
					found = true
					break
				}
			}
			if !found {
				continue
			}
		}
		
		// Event category filter
		if len(criteria.EventCategories) > 0 {
			found := false
			for _, category := range criteria.EventCategories {
				if msg.EventCategory == category {
					found = true
					break
				}
			}
			if !found {
				continue
			}
		}
		
		// Severity level filter
		if len(criteria.SeverityLevels) > 0 {
			found := false
			for _, severity := range criteria.SeverityLevels {
				if msg.Severity == severity {
					found = true
					break
				}
			}
			if !found {
				continue
			}
		}
		
		// Security events only
		if criteria.SecurityOnly && !msg.IsSecurity {
			continue
		}
		
		// Authentication events only
		if criteria.AuthOnly && !msg.IsAuthentication {
			continue
		}
		
		// Time range filter
		if !criteria.TimeRange.Start.IsZero() && msg.Timestamp.Before(criteria.TimeRange.Start) {
			continue
		}
		if !criteria.TimeRange.End.IsZero() && msg.Timestamp.After(criteria.TimeRange.End) {
			continue
		}
		
		// Threat name filter
		if criteria.HasThreatName && msg.ThreatName == "" {
			continue
		}
		
		// Username filter
		if criteria.HasUsername && msg.Username == "" {
			continue
		}
		
		// Source IP pattern filter
		if criteria.SourceIPPattern != "" {
			if matched, _ := regexp.MatchString(criteria.SourceIPPattern, msg.SourceIP); !matched {
				continue
			}
		}
		
		// Hostname pattern filter
		if criteria.HostnamePattern != "" {
			if matched, _ := regexp.MatchString(criteria.HostnamePattern, msg.Hostname); !matched {
				continue
			}
		}
		
		filtered = append(filtered, msg)
	}
	
	return filtered
}

// generateAggregateStats creates aggregate statistics from filtered messages
func (sc *SyslogCleaner) generateAggregateStats(messages []*NormalizedSyslogMessage, outputPath string) error {
	stats := struct {
		TotalMessages      int                    `json:"total_messages"`
		UniqueHosts        int                    `json:"unique_hosts"`
		UniqueSourceIPs    int                    `json:"unique_source_ips"`
		UniqueThreatNames  int                    `json:"unique_threat_names"`
		AvgRiskScore       float64                `json:"avg_risk_score"`
		TopHosts           []map[string]interface{} `json:"top_hosts"`
		TopSourceIPs       []map[string]interface{} `json:"top_source_ips"`
		TopThreatNames     []map[string]interface{} `json:"top_threat_names"`
		HourlyDistribution map[int]int            `json:"hourly_distribution"`
		DailyDistribution  map[string]int         `json:"daily_distribution"`
	}{
		HourlyDistribution: make(map[int]int),
		DailyDistribution:  make(map[string]int),
	}
	
	// Count occurrences
	hostCounts := make(map[string]int)
	ipCounts := make(map[string]int)
	threatCounts := make(map[string]int)
	riskScoreSum := 0
	
	for _, msg := range messages {
		hostCounts[msg.Hostname]++
		if msg.SourceIP != "" {
			ipCounts[msg.SourceIP]++
		}
		if msg.ThreatName != "" {
			threatCounts[msg.ThreatName]++
		}
		riskScoreSum += msg.RiskScore
		
		// Time-based statistics
		stats.HourlyDistribution[msg.Timestamp.Hour()]++
		day := msg.Timestamp.Format("2006-01-02")
		stats.DailyDistribution[day]++
	}
	
	stats.TotalMessages = len(messages)
	stats.UniqueHosts = len(hostCounts)
	stats.UniqueSourceIPs = len(ipCounts)
	stats.UniqueThreatNames = len(threatCounts)
	
	if len(messages) > 0 {
		stats.AvgRiskScore = float64(riskScoreSum) / float64(len(messages))
	}
	
	// Get top 10 hosts, IPs, and threats
	stats.TopHosts = getTopCounts(hostCounts, 10)
	stats.TopSourceIPs = getTopCounts(ipCounts, 10)
	stats.TopThreatNames = getTopCounts(threatCounts, 10)
	
	// Save statistics
	file, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("failed to create stats file: %v", err)
	}
	defer file.Close()
	
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	
	return encoder.Encode(stats)
}

// getTopCounts returns top N items from a count map
func getTopCounts(counts map[string]int, n int) []map[string]interface{} {
	type item struct {
		key   string
		count int
	}
	
	var items []item
	for k, v := range counts {
		items = append(items, item{k, v})
	}
	
	// Sort by count descending
	sort.Slice(items, func(i, j int) bool {
		return items[i].count > items[j].count
	})
	
	// Take top N
	if len(items) > n {
		items = items[:n]
	}
	
	var result []map[string]interface{}
	for _, item := range items {
		result = append(result, map[string]interface{}{
			"name":  item.key,
			"count": item.count,
		})
	}
	
	return result
}

// processDirectory processes all syslog files in a directory
func (sc *SyslogCleaner) processDirectory(inputDir, outputDir string) error {
	// Create output directory
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %v", err)
	}
	
	// Find all .log files
	pattern := filepath.Join(inputDir, "*.log")
	files, err := filepath.Glob(pattern)
	if err != nil {
		return fmt.Errorf("failed to find log files: %v", err)
	}
	
	if len(files) == 0 {
		return fmt.Errorf("no .log files found in %s", inputDir)
	}
	
	log.Printf("Found %d log files to process", len(files))
	
	var allMessages []*NormalizedSyslogMessage
	
	// Process each file
	for _, file := range files {
		messages, err := sc.processFile(file)
		if err != nil {
			log.Printf("Error processing file %s: %v", file, err)
			continue
		}
		
		allMessages = append(allMessages, messages...)
		
		// Save individual file results
		baseName := strings.TrimSuffix(filepath.Base(file), ".log")
		
		// Save as JSON
		jsonPath := filepath.Join(outputDir, baseName+"_cleaned.json")
		if err := sc.saveAsJSON(messages, jsonPath); err != nil {
			log.Printf("Error saving JSON for %s: %v", file, err)
		}
		
		// Save as CSV
		csvPath := filepath.Join(outputDir, baseName+"_cleaned.csv")
		if err := sc.saveAsCSV(messages, csvPath); err != nil {
			log.Printf("Error saving CSV for %s: %v", file, err)
		}
	}
	
	// Save combined results
	log.Printf("Saving combined results for %d total messages", len(allMessages))
	
	// Combined JSON
	combinedJSONPath := filepath.Join(outputDir, "all_cleaned.json")
	if err := sc.saveAsJSON(allMessages, combinedJSONPath); err != nil {
		return fmt.Errorf("failed to save combined JSON: %v", err)
	}
	
	// Combined CSV
	combinedCSVPath := filepath.Join(outputDir, "all_cleaned.csv")
	if err := sc.saveAsCSV(allMessages, combinedCSVPath); err != nil {
		return fmt.Errorf("failed to save combined CSV: %v", err)
	}
	
	// Normalized syslog format
	normalizedPath := filepath.Join(outputDir, "all_normalized.log")
	if err := sc.saveAsNormalizedSyslog(allMessages, normalizedPath); err != nil {
		return fmt.Errorf("failed to save normalized syslog: %v", err)
	}
	
	// Generate aggregate statistics
	statsPath := filepath.Join(outputDir, "aggregate_statistics.json")
	if err := sc.generateAggregateStats(allMessages, statsPath); err != nil {
		return fmt.Errorf("failed to generate aggregate stats: %v", err)
	}
	
	// Generate cleaning report
	if err := sc.generateReport(outputDir); err != nil {
		return fmt.Errorf("failed to generate cleaning report: %v", err)
	}
	
	// Save high-risk security events
	securityCriteria := FilterCriteria{
		MinRiskScore: 50,
		SecurityOnly: true,
	}
	securityEvents := sc.filterMessages(allMessages, securityCriteria)
	
	securityPath := filepath.Join(outputDir, "high_risk_security_events.json")
	if err := sc.saveAsJSON(securityEvents, securityPath); err != nil {
		log.Printf("Error saving security events: %v", err)
	}
	
	// Save authentication events
	authCriteria := FilterCriteria{
		AuthOnly: true,
	}
	authEvents := sc.filterMessages(allMessages, authCriteria)
	
	authPath := filepath.Join(outputDir, "authentication_events.json")
	if err := sc.saveAsJSON(authEvents, authPath); err != nil {
		log.Printf("Error saving authentication events: %v", err)
	}
	
	return nil
}

// createSampleConfig creates a sample configuration file
func createSampleConfig(configPath string) error {
	config := `# Syslog Cleaner Configuration
# Input and output settings
input_directory: "./syslog_output"
output_directory: "./cleaned_output"

# Processing options
remove_duplicates: true
normalize_timestamps: true
extract_fields: true
calculate_risk_scores: true

# Filtering options
filters:
  min_risk_score: 0
  max_risk_score: 100
  device_types: []  # Empty means all types
  event_categories: []  # Empty means all categories
  security_only: false
  auth_only: false

# Output formats
output_formats:
  json: true
  csv: true
  normalized_syslog: true
  
# Field extraction settings
extract_ips: true
extract_ports: true
extract_usernames: true
extract_interfaces: true
extract_threat_names: true
extract_bgp_info: true
extract_ospf_info: true

# Quality checks
data_quality_checks:
  validate_timestamps: true
  validate_priorities: true
  validate_hostnames: true
  validate_ip_addresses: true
  check_message_length: true
  
# Performance settings
batch_size: 1000
progress_interval: 1000
`
	
	return os.WriteFile(configPath, []byte(config), 0644)
}

func main() {
	fmt.Println("=== Syslog Data Cleaner and Normalizer ===")
	fmt.Println("Processing and cleaning syslog messages...")
	
	// Configuration
	inputDir := "./syslog_output"
	outputDir := "./cleaned_output"
	
	// Check if input directory exists
	if _, err := os.Stat(inputDir); os.IsNotExist(err) {
		log.Fatalf("Input directory does not exist: %s", inputDir)
	}
	
	// Create sample configuration
	configPath := "syslog_cleaner_config.yaml"
	if err := createSampleConfig(configPath); err != nil {
		log.Printf("Warning: Could not create sample config: %v", err)
	} else {
		log.Printf("Sample configuration created: %s", configPath)
	}
	
	// Initialize cleaner
	cleaner := NewSyslogCleaner()
	
	// Process all files
	startTime := time.Now()
	if err := cleaner.processDirectory(inputDir, outputDir); err != nil {
		log.Fatalf("Processing failed: %v", err)
	}
	
	duration := time.Since(startTime)
	
	// Print summary
	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("CLEANING COMPLETE!")
	fmt.Println(strings.Repeat("=", 60))
	fmt.Printf("Processing time: %v\n", duration)
	fmt.Printf("Total messages processed: %d\n", cleaner.stats.TotalProcessed)
	fmt.Printf("Valid messages: %d\n", cleaner.stats.ValidMessages)
	fmt.Printf("Invalid messages: %d\n", cleaner.stats.InvalidMessages)
	fmt.Printf("Duplicates removed: %d\n", cleaner.stats.DuplicatesRemoved)
	fmt.Printf("Data quality issues found: %d\n", 
		cleaner.stats.MalformedTimestamps+cleaner.stats.InvalidPriorities+cleaner.stats.MissingHostnames)
	
	fmt.Printf("\nOutput directory: %s\n", outputDir)
	fmt.Println("\nGenerated files:")
	fmt.Println("  - all_cleaned.json              (All messages in JSON format)")
	fmt.Println("  - all_cleaned.csv               (All messages in CSV format)")
	fmt.Println("  - all_normalized.log            (Normalized syslog format)")
	fmt.Println("  - high_risk_security_events.json (Security events with risk score ≥ 50)")
	fmt.Println("  - authentication_events.json    (All authentication events)")
	fmt.Println("  - aggregate_statistics.json     (Comprehensive statistics)")
	fmt.Println("  - cleaning_report.json          (Detailed cleaning report)")
	fmt.Println("  - cleaning_summary.txt          (Human-readable summary)")
	fmt.Println("  - *_cleaned.json                (Individual file results)")
	fmt.Println("  - *_cleaned.csv                 (Individual file results)")
	
	fmt.Println("\nData enhancements added:")
	fmt.Println("  ✓ Normalized timestamps to UTC")
	fmt.Println("  ✓ Classified device types")
	fmt.Println("  ✓ Categorized event types")
	fmt.Println("  ✓ Extracted IP addresses and ports")
	fmt.Println("  ✓ Identified usernames and interfaces")
	fmt.Println("  ✓ Extracted threat names and network info")
	fmt.Println("  ✓ Calculated risk scores")
	fmt.Println("  ✓ Added security and network event flags")
	fmt.Println("  ✓ Removed duplicates and invalid entries")
	
	fmt.Println("\nUse cleaned data for:")
	fmt.Println("  • SIEM system ingestion")
	fmt.Println("  • Machine learning model training")
	fmt.Println("  • Security analytics and threat hunting")
	fmt.Println("  • Network operations dashboards")
	fmt.Println("  • Compliance reporting and auditing")
	fmt.Println("  • Performance analysis and optimization")
	
	fmt.Printf("\nConfiguration saved to: %s\n", configPath)
}