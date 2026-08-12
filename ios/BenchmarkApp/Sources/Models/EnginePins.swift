import Foundation

/// Engine build identity per arm, bundled into the app at build time.
///
/// `scripts/stamp_engine_pins.sh` (a pre-build phase on both targets) reads the OBSERVED
/// vendored state — git describe of each Vendored/ clone, the CLiteRTLM binaryTarget
/// zip + checksum from LiteRT-LM's Package.swift, the sidecar tag written next to
/// llama.xcframework at download — into `Vendored/engine-pins.json`, which the app
/// target ships as a bundle resource. `BenchmarkRunner` copies the pin for the arm
/// under test into every `BenchmarkResult` as `engineVersion` / `engineArtifact`
/// (schema v1), which is what lets release-regression tracking join rows by engine
/// build instead of prose READMEs.
///
/// A resource, NOT an Info.plist key: the first design mutated the processed plist
/// post-build, and that undeclared edit raced the build graph — on incremental builds
/// it landed after ProcessInfoPlistFile (key dropped) or after codesign (install
/// rejected, 0xe8008001). Both observed 2026-08-13.
///
/// The Mac yardstick CLI bundles no resources: point `BENCH_ENGINE_PINS_FILE` at
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
        var path = Bundle.main.path(forResource: "engine-pins", ofType: "json")
        if path == nil {
            path = ProcessInfo.processInfo.environment["BENCH_ENGINE_PINS_FILE"]
        }
        if let path,
           let data = FileManager.default.contents(atPath: path),
           let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: [String: String]] {
            return obj
        }
        return [:]
    }()
}
