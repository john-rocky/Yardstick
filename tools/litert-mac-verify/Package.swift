// swift-tools-version:5.9
// litert-mac-verify — the Mac GSM8K quality instrument (the "yardstick" surface).
//
// This is the exact binary surface behind the published LiteRT GSM8K rows
// (v0.13.1 86.0; v0.15.0 88.0 at ctx2048; 89.0/92.0 off/thinking at ctx4096,
// captured 2026-08-04 — results/raw/2026-08-04-litert-0150-yardstick-thinking/).
// scripts/parity_gsm8k.py (--which int4) shells out to the release build:
//
//   swift build --package-path tools/litert-mac-verify -c release
//
// swift-litert-lm is pinned by EXACT revision, not a branch, for the same reason
// mlx-swift-lm is pinned in the top-level Package.swift: the instrument must not
// drift under the published numbers. The pinned commit (tag yardstick-2026-08-04)
// is the fork's v0.15.0 state: official release binaryTargets by checksum +
// the official Swift wrapper re-vendored at the tag (ThinkingConfig) + the
// LiteRTChat(thinking:) passthrough. That package carries no unsafe flags, so
// (unlike ios/BenchmarkApp/Vendored/LiteRT-LM) it is consumable as a remote
// dependency. Keep this pin in sync with environment.lock.json.
import PackageDescription

let package = Package(
  name: "litert-mac-verify",
  platforms: [.macOS(.v13)],
  dependencies: [
    .package(
      url: "https://github.com/john-rocky/swift-litert-lm.git",
      revision: "e4e48d9d8df9abfa0d1180b8c0305a3b0d16d25d")
  ],
  targets: [
    .executableTarget(
      name: "litert-mac-verify",
      dependencies: [
        .product(name: "LiteRTFoundation", package: "swift-litert-lm"),
        .product(name: "LiteRTLM", package: "swift-litert-lm")
      ]
    )
  ]
)
