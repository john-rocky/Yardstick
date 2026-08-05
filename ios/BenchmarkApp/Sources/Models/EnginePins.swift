import Foundation

/// Engine build identity per arm, stamped into the app at build time.
///
/// `scripts/stamp_engine_pins.sh` (a build phase on both targets) reads the OBSERVED
/// vendored state — git describe of each Vendored/ clone, the CLiteRTLM binaryTarget
/// zip + checksum from LiteRT-LM's Package.swift, the sidecar tag written next to
/// llama.xcframework at download — and writes it under the `BenchEnginePins`
/// Info.plist key. `BenchmarkRunner` copies the pin for the arm under test into every
/// `BenchmarkResult` as `engineVersion` / `engineArtifact` (schema v1), which is what
/// lets release-regression tracking join rows by engine build instead of prose READMEs.
///
/// The Mac yardstick CLI has no Info.plist: point `BENCH_ENGINE_PINS_FILE` at
/// `ios/BenchmarkApp/Vendored/engine-pins.json` (written by the same script).
/// Missing pin ⇒ `nil` ⇒ the row honestly records nothing — never a guessed default.
public enum EnginePins {
    public static func version(for kind: RuntimeKind) -> String? {
        pins[kind.rawValue]?["version"]
    }

    public static func artifact(for kind: RuntimeKind) -> String? {
        pins[kind.rawValue]?["artifact"]
    }

    private static let pins: [String: [String: String]] = {
        if let dict = Bundle.main.object(forInfoDictionaryKey: "BenchEnginePins")
            as? [String: [String: String]] {
            return dict
        }
        if let path = ProcessInfo.processInfo.environment["BENCH_ENGINE_PINS_FILE"],
           let data = FileManager.default.contents(atPath: path),
           let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: [String: String]] {
            return obj
        }
        return [:]
    }()
}
