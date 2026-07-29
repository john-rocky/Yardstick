// swift-tools-version: 6.0
// Yardstick — Apple Silicon AI benchmark harness (Mac + iPhone + iPad)

import PackageDescription

let package = Package(
    name: "Yardstick",
    platforms: [
        .iOS(.v18),
        .macOS(.v14),
    ],
    products: [
        // The bench harness as a reusable library — same code drives the
        // iOS BenchmarkApp and the macOS `yardstick` CLI.
        .library(name: "YardstickKit", targets: ["YardstickKit"]),

        // macOS command-line runner.
        .executable(name: "yardstick", targets: ["YardstickCLI"]),
    ],
    dependencies: [
        // Runtime SDKs — keep this list aligned with ios/BenchmarkApp/project.yml.
        // Phase 1 wires the MLX backend only; other runtimes are added in
        // follow-up commits once their Mac toolchain is verified.
        // Pinned to the EXACT revision the iOS app resolved (xcworkspace Package.resolved),
        // not `branch: "main"`: the two harnesses must share the model loader, and an
        // unpinned branch let them drift — measured 2026-07-28, the Mac resolution
        // (5b7e543) failed to load the same two checkpoints (PTQ per_layer_model_projection
        // shape, OptiQ KV-shared k_norm) that the iOS resolution (60bd0d7) loads and that
        // the 2026-07-27 iPhone cells were measured on. mlx-swift is pinned alongside it
        // for the same reason.
        .package(url: "https://github.com/ml-explore/mlx-swift-lm",
                 revision: "60bd0d7880c82980f9481f8be78862e9b63c58a3"),
        .package(url: "https://github.com/ml-explore/mlx-swift",
                 revision: "0bb916c67f4b9e5c682cbe02a42c701c93ab5021"),
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.8.1"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.0.0"),
        // LiteRT-LM ships a binary xcframework. SwiftPM rejects its
        // `.unsafeFlags(["-Xlinker","-all_load"])` from a versioned/remote product, so
        // (like the iOS project) depend on the LOCAL vendored clone — local path packages
        // are allowed to carry unsafe flags. `bootstrap.sh` clones it to this path
        // (GIT_LFS_SKIP_SMUDGE=1, skipping a broken upstream Android LFS object); the
        // iOS/macOS xcframeworks are remote binaryTargets fetched by SwiftPM.
        .package(path: "ios/BenchmarkApp/Vendored/LiteRT-LM"),
    ],
    targets: [
        .target(
            name: "YardstickKit",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Transformers", package: "swift-transformers"),
                // LiteRT-LM (`MediaPipeRuntime.swift`) — the adapter is
                // `#if canImport(LiteRTLM)`-gated, so it lights up here.
                .product(name: "LiteRTLM", package: "LiteRT-LM"),
            ],
            path: "ios/BenchmarkApp/Sources",
            // Keep Benchmark/, Models/, the MLX-related runtime sources
            // (LLMRuntime protocol, HFDownloader helper, MLXBridges, MLXRuntime)
            // and the LiteRT-LM adapter (clean SPM xcframework). The remaining
            // adapter files are excluded until their Mac build path is wired up
            // (each pulls in a different vendored SDK: AnemllCore,
            // llama.xcframework, CoreMLLLM, executorch).
            exclude: [
                "Info.plist",
                "Assets.xcassets",
                "Resources",
                "Views",
                "BenchmarkApp.swift",
                "Runtimes/CoreMLRuntime.swift",
                "Runtimes/AnemllRuntime.swift",
                "Runtimes/LlamaCppRuntime.swift",
                "Runtimes/ExecuTorchRuntime.swift",
                // Apple Core AI (CoreAILM) — iOS/macOS 27 only; the iOS app
                // wires it via XcodeGen. Excluded here like the other
                // vendor-SDK adapters so the macOS YardstickKit lib still builds.
                "Runtimes/CoreAIRuntime.swift",
                // VLM / camera adapters pull MLXVLM + AVFoundation (UI-only); the
                // text-LLM CLI doesn't need them, and MLXVLM isn't a CLI dependency.
                "Runtimes/MLXVLMRuntime.swift",
                "Runtimes/VLMRuntime.swift",
                "Runtimes/CoreMLVLMRuntime.swift",
                "Benchmark/CameraVLMTask.swift",
                "Models/VLMRunResult.swift",
                "Vision",
            ]
        ),

        .executableTarget(
            name: "YardstickCLI",
            dependencies: ["YardstickKit"],
            path: "apple/YardstickCLI/Sources",
            // The SwiftPM macOS CLI ships only the adapters wired for it (MLX +
            // LiteRT-LM + Apple FM); gate out the xcodebuild-only runtimes.
            swiftSettings: [.define("YARDSTICK_SPM")]
        ),
    ]
)
