import Foundation
#if canImport(CoreAILanguageModels)
import CoreAILanguageModels
import Metal
#endif
#if canImport(Tokenizers)
import Tokenizers
#endif

/// Apple **Core AI** adapter — the Core ML successor announced at WWDC 2026
/// (iOS / macOS 27). Loads a `.aimodel` LLM bundle produced by the official
/// `coreai.llm.export` pipeline and runs it through the official
/// `coreai-models` Swift runtime (`CoreAILM`), faithful to Apple's intended
/// on-device usage.
///
/// We deliberately use the low-level `EngineFactory` / `InferenceEngine` path —
/// the same one Apple's own `llm-benchmark` CLI tool uses
/// (`swift/Sources/Tools/benchmark/BenchmarkMain.swift`) — rather than the
/// high-level `LanguageModelSession`: it yields a raw token stream, so we get
/// true per-token timing (TTFT, inter-token latency) instead of a single
/// aggregate.
///
/// **Two compute paths, two bundles.** On iPhone the compute unit is decided by
/// the *export shape*, not just a runtime flag: the static iOS export
/// (`--platform iOS`) is detected as a chunked-static model → the `static-shape`
/// **ANE** engine; the dynamic export → the `coreai-pipelined` **GPU** engine.
/// So `…-ane` and `…-gpu` are two separate AOT-compiled bundles, distinguished
/// in the result rows with no schema change.
///
/// **iOS needs AOT compilation.** An exported `.aimodel` ships MLIR IR which
/// iOS cannot JIT; it must be compiled with `xcrun coreai-build compile
/// --platform iOS` to a `.aimodelc`, then `metadata.json assets.main` points at
/// the device-arch compiled file. See `methodology/coreai-ios.md` /
/// `scripts/bench_coreai_iphone.sh`.
///
/// The compiled bundles are **side-loaded** under `Documents/CoreAIModels/<name>/`
/// (large; not published to HF).
///
/// Requires iOS 27 / macOS 27 — the `coreai-models` Swift package floor. When
/// that package is not linked into the build (`canImport` false), this file
/// compiles to an unavailable stub so the rest of the app is unaffected.
public final class CoreAIRuntime: LLMRuntime, @unchecked Sendable {
    public let kind: RuntimeKind = .coreAI
    #if canImport(CoreAILanguageModels)
    public let isAvailable: Bool = true
    #else
    public let isAvailable: Bool = false
    #endif
    public let supportedModels: [ModelInfo] = ModelCatalog.coreAI

    nonisolated(unsafe) private var _loadedModelId: String?
    public var loadedModelId: String? { _loadedModelId }

    #if canImport(CoreAILanguageModels)
    nonisolated(unsafe) private var engine: (any InferenceEngine)?
    nonisolated(unsafe) private var tokenizer: (any Tokenizer)?
    nonisolated(unsafe) private var eosTokenIds: Set<Int32> = []
    #endif

    public init() {}

    // MARK: - Model id → bundle + compute variant

    /// Map a catalog id to its AOT-compiled bundle folder + the engine variant.
    /// The ANE bundle is the static iOS export compiled `--preferred-compute
    /// neural-engine` (structure → `static-shape`); the GPU bundle is the
    /// dynamic export compiled `--preferred-compute gpu` (structure →
    /// `coreai-pipelined`). The forced variant matches the bundle's structure;
    /// passing `nil` would auto-resolve to the same engine.
    private static func bundleSpec(for id: String) -> (folder: String, variant: String?)? {
        switch id {
        case "core-ai/qwen3-0.6b-ane": return ("qwen3_0_6b_ane", "static-shape")
        // June-lineage bundles (0.4.0-era compiles) kept as separate cells so cold numbers
        // can be compared 1:1 against the June sessions (artifact-lineage control).
        case "core-ai/qwen3-0.6b-ane-june": return ("qwen3_0_6b_ane_june", "static-shape")
        case "core-ai/qwen3-1.7b-gpu-june": return ("qwen3_1_7b_gpu_june", "coreai-pipelined")
        case "core-ai/qwen3-0.6b-gpu": return ("qwen3_0_6b_gpu", "coreai-pipelined")
        case "core-ai/qwen3-1.7b-ane": return ("qwen3_1_7b_ane", "static-shape")
        case "core-ai/qwen3-1.7b-gpu": return ("qwen3_1_7b_gpu", "coreai-pipelined")
        case "core-ai/qwen3-4b-ane":   return ("qwen3_4b_ane", "static-shape")
        case "core-ai/qwen3-4b-gpu":   return ("qwen3_4b_gpu", "coreai-pipelined")
        case "core-ai/qwen3-8b-ane":   return ("qwen3_8b_ane", "static-shape")
        case "core-ai/qwen3-8b-gpu":   return ("qwen3_8b_gpu", "coreai-pipelined")
        case "core-ai/deepseek-r1-1.5b-ane": return ("deepseek_r1_1_5b_ane", "static-shape")
        case "core-ai/deepseek-r1-1.5b-gpu": return ("deepseek_r1_1_5b_gpu", "coreai-pipelined")
        case "core-ai/lfm2.5-1.2b-gpu":  return ("lfm25_1_2b_gpu", "coreai-pipelined")
        case "core-ai/minicpm5-1b-gpu":  return ("minicpm5_1b_gpu", "coreai-pipelined")
        case "core-ai/tinyswallow-1.5b-ane": return ("tinyswallow_1_5b_ane", "static-shape")
        case "core-ai/tinyswallow-1.5b-gpu": return ("tinyswallow_1_5b_gpu", "coreai-pipelined")
        case "core-ai/vibethinker-1.5b-ane": return ("vibethinker_1_5b_ane", "static-shape")
        case "core-ai/vibethinker-1.5b-gpu": return ("vibethinker_1_5b_gpu", "coreai-pipelined")
        // 2026-06-25 export pass — GPU for all 6, ANE for llama/olmo2/smollm3 (ministral/gemma3/phi ANE pending)
        case "core-ai/ministral-3b-gpu":  return ("ministral3_3b_gpu", "coreai-pipelined")
        case "core-ai/gemma3-1b-gpu":     return ("gemma3_1b_gpu", "coreai-pipelined")
        // Gemma 4 E4B (Per-Layer-Embeddings). The `_tbl` decode graph gathers the PLE in-graph
        // from a mmap'd static table — the bundle folder also carries `ple/embed_per_layer.i8`
        // + `.scale.f32`, wired as EngineOptions.staticInputBuffers (see loadModel / GemmaPLEBench).
        case "core-ai/gemma4-e4b-gpu":    return ("gemma4_e4b_gpu", "coreai-pipelined")
        // The 2026-07-14 "EngineFactory wall" was a MISDIAGNOSIS — root-caused 2026-07-18 on
        // this app: the engine loads and generates fine through EngineFactory; what fataled
        // was our own warmup(queryLength: 8) after createEngine (S=1-only graph → binary-layer
        // NDArrayDescriptor fatal that `try?` can't catch). Fixed by skipping warmup for
        // gemma4_* (see loadModel). Two further requirements, both data-side: the bundle's
        // tokenizer_config.json must carry a chat_template (gemma-4 ships it as a separate
        // chat_template.jinja that swift-transformers doesn't read → raw-encode → degenerate
        // "<turn|>" output), and COREAI_CHUNK_THRESHOLD=1 must be set early (BenchmarkApp.init).
        case "core-ai/gemma4-e2b-gpu":    return ("gemma4_e2b_gpu", "coreai-pipelined")
        case "core-ai/phi-4-mini-gpu":    return ("phi4_mini_gpu", "coreai-pipelined")
        case "core-ai/llama-3.2-3b-ane":  return ("llama32_3b_ane", "static-shape")
        case "core-ai/llama-3.2-3b-gpu":  return ("llama32_3b_gpu", "coreai-pipelined")
        case "core-ai/olmo2-1b-ane":      return ("olmo2_1b_ane", "static-shape")
        case "core-ai/olmo2-1b-gpu":      return ("olmo2_1b_gpu", "coreai-pipelined")
        case "core-ai/smollm3-3b-ane":    return ("smollm3_3b_ane", "static-shape")
        case "core-ai/smollm3-3b-gpu":    return ("smollm3_3b_gpu", "coreai-pipelined")
        // 2026-06-26 static-GPU experiment — static-shape structure (extend_* fns, palettized LUT, identical to the
        // *_ane bundle) AOT-compiled `--preferred-compute gpu` → 0 ANE regions. Same "static-shape" engine as the ANE
        // bundle (structure match); the bundle's 0-ANE placement runs it on GPU. Forms the 3-way with *_ane (static/ANE)
        // and *_gpu (dynamic/GPU): static-ANE vs static-GPU = pure engine; static-GPU vs dynamic-GPU = shape/cold.
        // static_gpu = GPU MPSGraph (0 ANE regions) export. ⚠ 2026-06-26: these bundles compile but DO NOT LOAD in any
        // engine variant (static-shape / coreai-pipelined / nil=auto all fail EngineFactory POSIX Code=2 "No such file"):
        // the GPU compile emits an `mpsExecutable.mpsgraphpackage` (original_model/specialized_model/resources.bin) that
        // lacks the per-bucket `binary_0.llir.bundle/.../extend_*` artifacts the static engine needs. Needs an export-side
        // re-compile (full GPU bucket specialization / gemma4-bucketed port) — deferred to a separate session. Entries
        // kept wired so that session only needs to drop in loadable bundles. (nil = let the factory auto-resolve.)
        case "core-ai/deepseek-r1-1.5b-static-gpu": return ("deepseek_r1_1_5b_static_gpu", nil)
        case "core-ai/tinyswallow-1.5b-static-gpu": return ("tinyswallow_1_5b_static_gpu", nil)
        case "core-ai/vibethinker-1.5b-static-gpu": return ("vibethinker_1_5b_static_gpu", nil)
        case "core-ai/qwen3-0.6b-static-gpu":       return ("qwen3_0_6b_static_gpu", nil)
        case "core-ai/qwen3-1.7b-static-gpu":       return ("qwen3_1_7b_static_gpu", nil)
        case "core-ai/qwen3-4b-static-gpu":         return ("qwen3_4b_static_gpu", nil)
        case "core-ai/qwen3-8b-static-gpu":         return ("qwen3_8b_static_gpu", nil)
        case "core-ai/olmo2-1b-static-gpu":         return ("olmo2_1b_static_gpu", nil)
        case "core-ai/smollm3-3b-static-gpu":       return ("smollm3_3b_static_gpu", nil)
        case "core-ai/llama-3.2-3b-static-gpu":     return ("llama32_3b_static_gpu", nil)
        default:                       return nil
        }
    }

    /// Resolve a side-loaded `.aimodel` bundle folder on device. We look in
    /// `Documents/CoreAIModels/<folder>/` first (push it there with
    /// `xcrun devicectl device copy to …` or Finder file sharing), then fall
    /// back to an embedded app-bundle resource.
    private static func resolveBundleURL(folder: String) -> URL? {
        let fm = FileManager.default
        if let docs = fm.urls(for: .documentDirectory, in: .userDomainMask).first {
            let u = docs.appendingPathComponent("CoreAIModels/\(folder)", isDirectory: true)
            if fm.fileExists(atPath: u.appendingPathComponent("metadata.json").path) { return u }
        }
        if let res = Bundle.main.url(forResource: folder, withExtension: nil),
           fm.fileExists(atPath: res.appendingPathComponent("metadata.json").path) {
            return res
        }
        return nil
    }

    #if canImport(CoreAILanguageModels)
    /// True when the bundle side-loads PLE tables — i.e. it can only run on a patched engine.
    /// Cheap file check, so it compiles against the stock runtime too.
    private static func hasPLETables(bundleURL: URL) -> Bool {
        FileManager.default.fileExists(
            atPath: bundleURL.appendingPathComponent("ple/embed_per_layer.i8").path)
    }

    #if COREAI_STATIC_INPUTS
    /// Gemma-4 E-series (E2B/E4B) carry Per-Layer-Embeddings whose table is too large to live
    /// in the graph. The `_tbl` decode graph gathers it in-graph from two static inputs
    /// (`ple_table` = int8 rows, `ple_scale` = per-row f32), mmap'd no-copy and bound on every
    /// encode. We side-load them next to the bundle under `ple/`. Returns [:] for non-PLE models.
    /// Adapted from `~/code/coreai/ondevice/GemmaPLEBench` (the reference that benched E4B on device).
    ///
    /// Requires the patched engine: `StaticInputBuffer` / `EngineOptions.staticInputBuffers` are
    /// not part of Apple's released coreai-models.
    private static func staticPLEBuffers(bundleURL: URL) -> [String: StaticInputBuffer] {
        let pleDir = bundleURL.appendingPathComponent("ple", isDirectory: true)
        let files = ["ple_table": "embed_per_layer.i8", "ple_scale": "embed_per_layer.scale.f32"]
        let fm = FileManager.default
        guard fm.fileExists(atPath: pleDir.appendingPathComponent(files["ple_table"]!).path),
              let device = MTLCreateSystemDefaultDevice() else { return [:] }
        var out: [String: StaticInputBuffer] = [:]
        for (name, file) in files {
            guard let buf = Self.mapTableBuffer(url: pleDir.appendingPathComponent(file), device: device)
            else { return [:] }
            out[name] = StaticInputBuffer(buf)
        }
        return out
    }

    /// mmap a table file read-only and wrap it as a no-copy, page-aligned MTLBuffer. The engine
    /// binds it unchanged on every encode and never writes it; COW pages stay clean/evictable so
    /// the multi-GB table doesn't count as dirty-resident (this is what lets E4B fit on device).
    private static func mapTableBuffer(url: URL, device: any MTLDevice) -> (any MTLBuffer)? {
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else { return nil }
        defer { close(fd) }
        let size = Int(lseek(fd, 0, SEEK_END))
        guard size > 0 else { return nil }
        let page = Int(getpagesize())
        let mapLen = (size + page - 1) / page * page
        guard let p = mmap(nil, mapLen, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0),
              p != MAP_FAILED else { return nil }
        return device.makeBuffer(bytesNoCopy: UnsafeMutableRawPointer(mutating: p),
                                 length: mapLen, options: .storageModeShared, deallocator: nil)
    }
    #endif  // COREAI_STATIC_INPUTS
    #endif  // canImport(CoreAILanguageModels)

    // MARK: - Load

    public func loadModel(
        _ model: ModelInfo,
        progress: @Sendable @escaping (Double) -> Void
    ) async throws {
        guard supportedModels.contains(where: { $0.id == model.id }) else {
            throw LLMRuntimeError.modelNotInCatalog(model.id)
        }
        #if canImport(CoreAILanguageModels)
        guard let spec = Self.bundleSpec(for: model.id) else {
            throw LLMRuntimeError.loadFailed("Unknown Core AI model id \(model.id).")
        }
        guard let bundleURL = Self.resolveBundleURL(folder: spec.folder) else {
            throw LLMRuntimeError.loadFailed(
                "Core AI bundle '\(spec.folder)' not found. Side-load the exported "
                + "folder into Documents/CoreAIModels/\(spec.folder)/ "
                + "(it must contain metadata.json, the .aimodel, and tokenizer/)."
            )
        }

        var step = "start"
        do {
            progress(0.15)
            // Mirror Apple's llm-benchmark tool: build a ModelConfig from the
            // LanguageBundle and hand it to EngineFactory.
            step = "LanguageBundle(\(bundleURL.lastPathComponent))"
            let bundle = try LanguageBundle(at: bundleURL)
            step = "requireModelURL"
            let modelURL = try bundle.requireModelURL(for: ModelBundle.ComponentKey.main)
            step = "ModelConfig"
            let engineConfig = ModelConfig(
                name: bundle.name,
                tokenizer: bundle.tokenizer,
                vocabSize: bundle.vocabSize,
                maxContextLength: bundle.maxContextLength,
                serializedModel: [bundle.modelAssetPath],
                function: bundle.language.functionMap?.name(for: "main") ?? "main"
            )
            let configData = try JSONEncoder().encode(engineConfig)
            progress(0.35)

            // Per-Layer-Embedding models (Gemma-4 E-series) bind the PLE table as static inputs;
            // the in-graph gather needs S=1 prefill steps (COREAI_CHUNK_THRESHOLD=1), set before
            // engine creation. Empty for non-PLE models (no behaviour change).
            //
            // COREAI_STATIC_INPUTS gates this: EngineOptions.staticInputBuffers is NOT in Apple's
            // released coreai-models (absent from 0.1.0 and 0.2.0) — it is a local engine patch.
            // A stock clone therefore builds every arm except this path, and PLE models (Gemma-4
            // E2B/E4B) are unavailable rather than the whole app failing to compile. See
            // methodology/core-ai-arm-provenance.md.
            // S=1 decode-only graphs need single-step prefill set BEFORE engine
            // creation, in the stock path too: Gemma-4 E-series, and the LFM2.5
            // ShortConv-hybrid export (its graph is `..._decode_...`; the zoo
            // runner prefill-steps it token-by-token — card: "prompt tok/s ≈
            // decode tok/s"). Chunked prefill fatals in NDArrayDescriptor
            // ("dimension 1 of 8 is not a valid substitution for source shape 1",
            // reproduced 2026-08-26 in results/raw/2026-08-26-iphone-coreai-pairs).
            let isSingleStep = spec.folder.hasPrefix("gemma4_") || spec.folder == "lfm25_1_2b_gpu"
            if isSingleStep {
                setenv("COREAI_CHUNK_THRESHOLD", "1", 1)
            }
            #if COREAI_STATIC_INPUTS
            let pleBuffers = Self.staticPLEBuffers(bundleURL: bundleURL)
            if !pleBuffers.isEmpty {
                setenv("COREAI_CHUNK_THRESHOLD", "1", 1)
            }
            step = "EngineFactory(variant=\(spec.variant ?? "auto"), model=\(modelURL.lastPathComponent), ple=\(pleBuffers.count))"
            let options = EngineOptions(
                variant: spec.variant, kvCacheStrategy: .auto, staticInputBuffers: pleBuffers)
            #else
            if Self.hasPLETables(bundleURL: bundleURL) {
                throw LLMRuntimeError.unsupported(
                    "\(model.id) is a per-layer-embedding model and needs EngineOptions.staticInputBuffers, "
                    + "which Apple's released coreai-models does not expose. Build with "
                    + "COREAI_STATIC_INPUTS against a patched engine to measure this arm.")
            }
            step = "EngineFactory(variant=\(spec.variant ?? "auto"), model=\(modelURL.lastPathComponent), ple=stock)"
            let options = EngineOptions(variant: spec.variant, kvCacheStrategy: .auto)
            #endif
            let engine = try await EngineFactory.createEngine(
                config: configData,
                modelURL: modelURL,
                options: options
            )
            progress(0.7)

            step = "loadTokenizer"
            let tok = try await bundle.loadTokenizer()
            var eos: Set<Int32> = []
            if let e = tok.eosTokenId { eos.insert(Int32(e)) }

            // Trigger kernel compilation up front so it folds into load time.
            // NOT for Gemma-4 PLE bundles: their decode graphs are S=1-only, and
            // warmup(queryLength: 8) fatals inside the binary runtime
            // ("NDArrayDescriptor.swift:139 ... dimension 1 of 8 is not a valid substitution
            // for source shape 1") — a fatalError, so `try?` cannot catch it. This was the
            // whole "EngineFactory wall": the engine loads fine, the warmup kills it.
            // For S=1 graphs the first generate step is the warmup (GemmaPLEDeviceBench rule:
            // never call engine.warmup on these). Root-caused in the other checkout 2026-07-18;
            // reproduced here 2026-07-27 and ported.
            step = "warmup"
            if !isSingleStep {
                try? await engine.warmup(queryLength: 8, sampling: SamplingConfiguration(temperature: 0))
            }

            self.engine = engine
            self.tokenizer = tok
            self.eosTokenIds = eos
            self._loadedModelId = model.id
            progress(1)
        } catch let e as LLMRuntimeError {
            throw e
        } catch {
            throw LLMRuntimeError.loadFailed("[\(step)] \(error)")
        }
        #else
        throw LLMRuntimeError.unsupported("Core AI runtime not present in this build (requires the coreai-models Swift package, iOS/macOS 27).")
        #endif
    }

    public func unloadModel() async {
        #if canImport(CoreAILanguageModels)
        engine = nil
        tokenizer = nil
        eosTokenIds = []
        #endif
        _loadedModelId = nil
    }

    // MARK: - Generate

    public func generate(
        prompt: String,
        parameters: GenerationParameters
    ) -> AsyncThrowingStream<GenerationEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await self.runGenerate(prompt: prompt, parameters: parameters, continuation: continuation)
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func runGenerate(
        prompt: String,
        parameters: GenerationParameters,
        continuation: AsyncThrowingStream<GenerationEvent, Error>.Continuation
    ) async throws {
        #if canImport(CoreAILanguageModels)
        guard let engine, let tokenizer else { throw LLMRuntimeError.modelNotLoaded }

        // Tokenize with the model's chat template (greedy, deterministic — the
        // same sampling Apple's benchmark tool uses: temperature 0).
        let messages: [[String: String]] = [["role": "user", "content": prompt]]
        let promptIds: [Int] = (try? tokenizer.applyChatTemplate(messages: messages))
            ?? tokenizer.encode(text: prompt)
        let inputIds = promptIds.map { Int32($0) }

        try await engine.reset()
        let sampling = SamplingConfiguration(temperature: 0)
        let options = InferenceOptions(maxTokens: parameters.maxTokens, includeLogits: false)

        let prefillStart = CFAbsoluteTimeGetCurrent()
        var firstTokenAt: CFAbsoluteTime?
        var genCount = 0
        var accumIds: [Int] = []
        var emitted = ""

        // `await`: generate() became async in coreai-models 0.2.0. Harmless against an older
        // sync engine (Swift only warns that no async work occurs), so the patched-engine build
        // still compiles.
        let stream = try await engine.generate(
            with: inputIds,
            samplingConfiguration: sampling,
            inferenceOptions: options
        )
        for try await out in stream {
            try Task.checkCancellation()
            if firstTokenAt == nil { firstTokenAt = CFAbsoluteTimeGetCurrent() }
            let tid = out.tokenId
            genCount += 1
            if eosTokenIds.contains(tid) { break }
            accumIds.append(Int(tid))
            // Incremental decode → emit only the new text so the runner gets
            // real per-token timing for inter-token-latency percentiles.
            // Diff by COMMON PREFIX, never by slicing `current` with an index
            // taken from `emitted`: a String.Index is only valid for the string
            // it came from, so `current[emitted.endIndex...]` is undefined and
            // can crash or corrupt on byte-level tokenizers where a multi-byte
            // character straddles two tokens (a partial "�" that resolves on the
            // next token). dropFirst(sharedCount) is index-safe for any tokenizer.
            let current = tokenizer.decode(tokens: accumIds)
            if current != emitted {
                let shared = current.commonPrefix(with: emitted).count
                if current.count > shared {
                    let delta = String(current.dropFirst(shared))
                    continuation.yield(.chunk(delta))
                }
                emitted = current
            }
        }

        let end = CFAbsoluteTimeGetCurrent()
        let promptTime = (firstTokenAt ?? end) - prefillStart
        let generateTime = max(end - (firstTokenAt ?? prefillStart), 0.001)
        continuation.yield(.info(GenerationInfo(
            promptTokenCount: inputIds.count,
            generationTokenCount: genCount,
            promptTime: promptTime,
            generateTime: generateTime,
            stopReason: genCount >= parameters.maxTokens ? .length : .stop
        )))
        continuation.finish()
        #else
        throw LLMRuntimeError.unsupported("Core AI runtime not present in this build.")
        #endif
    }
}
