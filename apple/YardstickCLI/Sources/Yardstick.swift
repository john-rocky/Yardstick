// Yardstick — Apple Silicon AI benchmark CLI (Mac).
//
// Usage:
//   yardstick run --task short-chat \
//                 --runtime mlx-swift \
//                 --model mlx-community/gemma-4-e2b-it-4bit \
//                 [--output results/raw/<auto>.jsonl]
//
// One run produces one `BenchmarkResult`. Multiple invocations append to the
// output file as JSONL. Aggregation lives outside the CLI (`results/` +
// scripts), so this binary stays tiny.

import Foundation

// The CLI is built two ways:
//
//   1. `swift build` from the repo root — `main.swift` is in the
//      `YardstickCLI` SPM target, and the bench types live in the
//      sibling `YardstickKit` library target.
//   2. `xcodebuild -scheme yardstick` — `main.swift` is compiled
//      together with `Sources/{Benchmark,Models,Runtimes}/...` into a
//      single `yardstick` Mac tool target. In that build there is no
//      separate `YardstickKit` module.
//
// The conditional import keeps both paths happy.
#if canImport(YardstickKit)
    import YardstickKit
#endif

@main
struct YardstickApp {
    static func main() async {
        do {
            try await runMain()
        } catch {
            FileHandle.standardError.write(Data("error: \(error)\n".utf8))
            exit(1)
        }
    }

    static func runMain() async throws {
        let argv = Array(CommandLine.arguments.dropFirst())
        guard let subcommand = argv.first else {
            printUsage()
            exit(2)
        }

        switch subcommand {
        case "run":
            try await runCommand(Array(argv.dropFirst()))
        case "list":
            listCatalog()
        case "--help", "-h", "help":
            printUsage()
        default:
            FileHandle.standardError.write(Data("unknown command: \(subcommand)\n".utf8))
            printUsage()
            exit(2)
        }
    }

    // MARK: - `yardstick run`

    static func runCommand(_ argv: [String]) async throws {
        var taskID = "short-chat"
        var runtimeID = "mlx-swift"
        var modelID: String? = nil
        var outputPath: String? = nil
        var coldRun = true
        var runs = 1

        var i = 0
        while i < argv.count {
            let arg = argv[i]
            switch arg {
            case "--task":
                taskID = argv.value(after: &i)
            case "--runtime":
                runtimeID = argv.value(after: &i)
            case "--model":
                modelID = argv.value(after: &i)
            case "--output":
                outputPath = argv.value(after: &i)
            case "--warm":
                coldRun = false
                i += 1
            case "--runs":
                runs = max(1, Int(argv.value(after: &i)) ?? 1)
            default:
                FileHandle.standardError.write(Data("unknown flag: \(arg)\n".utf8))
                exit(2)
            }
        }

        let runtime = try makeRuntime(id: runtimeID)
        let task = try makeTask(id: taskID)
        let model = try resolveModel(idOrHF: modelID, runtime: runtime)

        let runner = BenchmarkRunner()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601

        // `--runs N` mirrors the iOS app's fairness-rules §2 warm protocol: one
        // process, the model stays loaded across runs — run 1 is cold (fresh
        // process, weights on disk), runs 2..N are warm; report the median of
        // runs 2..N as warm. Each run gets its own JSONL record (coldRun flags
        // the split), identical to what the iPhone campaign driver imports.
        for runIndex in 1...runs {
            let config = BenchmarkRunner.Configuration(
                runtime: runtime,
                model: model,
                task: task,
                coldRun: runs > 1 ? (runIndex == 1) : coldRun
            )

            FileHandle.standardError.write(Data(
                "yardstick: run \(runIndex)/\(runs) task=\(taskID) runtime=\(runtimeID) model=\(model.id)\n".utf8
            ))

            let result = try await runner.run(config)
            let json = try encoder.encode(result)

            // Always print the result to stdout.
            FileHandle.standardOutput.write(json)
            FileHandle.standardOutput.write(Data("\n".utf8))

            // Optionally append to a JSONL file.
            if let outputPath {
                try appendJSONL(result: result, path: outputPath)
                FileHandle.standardError.write(Data("yardstick: appended to \(outputPath)\n".utf8))
            }

            // Friendly one-line summary on stderr.
            let m = result.metrics
            FileHandle.standardError.write(Data(
                "yardstick: run=\(runIndex) cold=\(runIndex == 1 && (runs > 1 || coldRun) ? 1 : 0) TTFT=\(m.firstTokenLatencyMS)ms decode=\(String(format: "%.2f", m.decodeTokensPerSecond))tok/s peakMB=\(Int(m.memoryPeakDuringDecodeMB))\n".utf8
            ))
        }
    }

    // MARK: - `yardstick list`

    static func listCatalog() {
        print("Available runtimes (Mac CLI):")
        print("  mlx-swift    — MLX Swift LM (default, MLX GPU)")
        print("  coreml-llm   — CoreML via swift-transformers / CoreMLLLM (ANE / GPU)")
        print("  executorch   — PyTorch ExecuTorch (XNNPACK / CoreML delegate)")
        print("  llama-cpp    — llama.cpp via vendored xcframework (CPU + Metal)")
        print("  anemll       — ANEMLL via vendored anemll-swift-cli (ANE)")
        print("  apple-fm     — Apple Foundation Models (macOS 26+, Apple-Intelligence-eligible Macs)")
        print("  litert-lm    — LiteRT-LM (google-ai-edge/LiteRT-LM, Metal GPU)")
        print("")
        print("Available tasks:")
        print("  short-chat   — 128-token reply, measures TTFT + decode tok/s")
        print("  long-context — 2K-token prefill + short reply, measures prefill")
        print("  cactus-parity— 1K-token prefill + 100-token reply, matches `cactus benchmark`")
        print("  sustained    — 512-token generation, watches thermal drift")
        print("  lifecycle    — short generation x N, mimics chat session reuse")
        print("")
        print("Available models per runtime — pass `--model <id-or-hf-repo>`.")
        for (label, models) in [
            ("mlx-swift", ModelCatalog.mlx),
            ("coreml-llm", ModelCatalog.coreML),
            ("executorch", ModelCatalog.executorch),
            ("llama-cpp", ModelCatalog.llamaCpp),
            ("anemll", ModelCatalog.anemll),
            ("apple-fm", ModelCatalog.appleFM),
            ("litert-lm", ModelCatalog.liteRTLM),
        ] {
            guard !models.isEmpty else { continue }
            print("\n  [\(label)]")
            for m in models {
                let size = m.onDiskSizeMB.map { "\($0) MB" } ?? "?"
                print("    \(m.id) — \(m.displayName) (\(m.quantization), ~\(size))")
            }
        }
    }

    // MARK: - Helpers

    static func makeRuntime(id: String) throws -> any LLMRuntime {
        switch id {
        case "mlx-swift", "mlx":
            return MLXRuntime()
        // These adapters pull vendored SDKs not yet wired into the SwiftPM macOS
        // build (CoreMLLLM / executorch / llama.xcframework / AnemllCore); the
        // xcodebuild target still ships them. YARDSTICK_SPM is defined only for the
        // SwiftPM CLI, where their sources are excluded from YardstickKit.
        #if !YARDSTICK_SPM
        case "coreml-llm", "coreml":
            return CoreMLRuntime()
        case "executorch", "et":
            return ExecuTorchRuntime()
        case "llama-cpp", "llama.cpp", "llamacpp":
            return LlamaCppRuntime()
        case "anemll":
            return AnemllRuntime()
        #endif
        case "apple-fm", "apple", "fm", "foundation-models":
            return AppleFMRuntime()
        case "litert-lm", "mediapipe":
            return MediaPipeRuntime()
        default:
            throw CLIError.invalidArgument(
                "unknown runtime '\(id)' — supported on Mac: mlx-swift, coreml-llm, executorch, llama-cpp, anemll, apple-fm, litert-lm"
            )
        }
    }

    static func makeTask(id: String) throws -> any BenchmarkTask {
        switch id {
        case "short-chat":
            return ShortChatTask()
        case "long-context":
            return LongContextTask()
        case "sustained":
            return SustainedGenerationTask()
        case "lifecycle":
            return AppLifecycleTask()
        default:
            // Resolve any other registered task by id (e.g. long-context-8k /
            // long-context-32k sweep variants, energy) from the shared catalog.
            if let t = BenchmarkTaskCatalog.task(for: id) { return t }
            throw CLIError.invalidArgument(
                "unknown task '\(id)' — see `yardstick list`"
            )
        }
    }

    static func resolveModel(idOrHF: String?, runtime: any LLMRuntime) throws -> ModelInfo {
        let supported = runtime.supportedModels
        guard !supported.isEmpty else {
            throw CLIError.invalidArgument(
                "runtime \(runtime.kind.displayName) has no supportedModels listed"
            )
        }
        guard let idOrHF else {
            return supported[0]
        }
        if let match = supported.first(where: { $0.id == idOrHF || $0.hfRepoId == idOrHF }) {
            return match
        }
        // Fall back: synthesize a ModelInfo for any HF repo id. The runtime
        // will fail loadModel if the repo doesn't fit its expected layout.
        return ModelInfo(
            id: idOrHF,
            displayName: idOrHF,
            quantization: "?",
            hfRepoId: idOrHF
        )
    }

    static func appendJSONL(result: BenchmarkResult, path: String) throws {
        let url = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let line = try encoder.encode(result) + Data("\n".utf8)

        if FileManager.default.fileExists(atPath: url.path) {
            let handle = try FileHandle(forWritingTo: url)
            try handle.seekToEnd()
            try handle.write(contentsOf: line)
            try handle.close()
        } else {
            try line.write(to: url)
        }
    }

    static func printUsage() {
        print(
            """
            Yardstick — Apple Silicon AI benchmark CLI

            Usage:
              yardstick run --task <id> --runtime <id> --model <id|hf-repo> [--output <path>] [--warm] [--runs N]
              yardstick list
              yardstick help

            Examples:
              yardstick run --task short-chat \\
                            --runtime mlx-swift \\
                            --model mlx-community/gemma-4-e2b-it-4bit

              yardstick run --task sustained \\
                            --runtime mlx-swift \\
                            --output results/raw/m4max-mlx-sustained.jsonl

            See `yardstick list` for the catalog of available tasks and models.
            """
        )
    }
}

enum CLIError: Error, CustomStringConvertible {
    case invalidArgument(String)

    var description: String {
        switch self {
        case .invalidArgument(let msg):
            return msg
        }
    }
}

private extension Array where Element == String {
    func value(after index: inout Int) -> String {
        guard index + 1 < count else {
            FileHandle.standardError.write(Data(
                "flag \(self[index]) requires a value\n".utf8
            ))
            exit(2)
        }
        defer { index += 2 }
        return self[index + 1]
    }
}
