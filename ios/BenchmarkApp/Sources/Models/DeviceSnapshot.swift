import Foundation
#if canImport(UIKit)
import UIKit
#endif

public struct DeviceSnapshot: Codable, Sendable {
    public let modelIdentifier: String
    public let systemName: String
    public let systemVersion: String
    public let processorCount: Int
    public let physicalMemoryMB: Int
    public let isLowPowerModeEnabled: Bool
    // var (not let) so the runner can refresh these to end-of-run values: a
    // launch-then-unplug energy run starts plugged but discharges mid-run, and
    // we want batteryState to reflect what actually happened.
    public var batteryState: String
    public var batteryLevel: Float
    public let initialThermalState: String
    public let buildConfiguration: String

    public static func capture() -> DeviceSnapshot {
        let info = ProcessInfo.processInfo
        let physicalMemory = Int(info.physicalMemory / (1024 * 1024))

        let battery = currentBattery()
        let batteryState = battery.state
        let batteryLevel = battery.level
        #if canImport(UIKit)
        let systemName = UIDevice.current.systemName
        let systemVersion = UIDevice.current.systemVersion
        #else
        let systemName = "macOS"
        let systemVersion = info.operatingSystemVersionString
        #endif

        return DeviceSnapshot(
            modelIdentifier: hardwareModelIdentifier(),
            systemName: systemName,
            systemVersion: systemVersion,
            processorCount: info.processorCount,
            physicalMemoryMB: physicalMemory,
            isLowPowerModeEnabled: info.isLowPowerModeEnabled,
            batteryState: batteryState,
            batteryLevel: batteryLevel,
            initialThermalState: ThermalMonitor.describe(info.thermalState),
            buildConfiguration: buildConfiguration()
        )
    }

    /// Reads the current battery state + level. Returns `("unknown", -1)` on
    /// platforms without UIKit (the Mac CLI). Used both for the start-of-run
    /// snapshot and to refresh the snapshot to end-of-run values.
    public static func currentBattery() -> (state: String, level: Float) {
        #if canImport(UIKit)
        UIDevice.current.isBatteryMonitoringEnabled = true
        let state: String
        switch UIDevice.current.batteryState {
        case .charging: state = "charging"
        case .full: state = "full"
        case .unplugged: state = "unplugged"
        case .unknown: state = "unknown"
        @unknown default: state = "unknown"
        }
        return (state, UIDevice.current.batteryLevel)
        #else
        return ("unknown", -1)
        #endif
    }

    private static func hardwareModelIdentifier() -> String {
        // utsname().machine is the device identifier on iOS ("iPhone18,1") but
        // just the ISA on macOS ("arm64" — every Mac pools into one row).
        // hw.model gives the Mac identifier ("Mac16,9").
        #if os(macOS)
        var size = 0
        sysctlbyname("hw.model", nil, &size, nil, 0)
        if size > 0 {
            var buf = [CChar](repeating: 0, count: size)
            if sysctlbyname("hw.model", &buf, &size, nil, 0) == 0 {
                let model = String(cString: buf)
                if !model.isEmpty { return model }
            }
        }
        #endif
        var systemInfo = utsname()
        uname(&systemInfo)
        let mirror = Mirror(reflecting: systemInfo.machine)
        let identifier = mirror.children.reduce(into: "") { partial, element in
            guard let value = element.value as? Int8, value != 0 else { return }
            partial.append(Character(UnicodeScalar(UInt8(value))))
        }
        return identifier.isEmpty ? "unknown" : identifier
    }

    private static func buildConfiguration() -> String {
        #if DEBUG
        return "Debug"
        #else
        return "Release"
        #endif
    }
}
