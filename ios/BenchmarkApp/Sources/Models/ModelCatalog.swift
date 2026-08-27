import Foundation

/// A model packaged for a specific runtime.
///
/// Different runtimes need different artifact formats — GGUF for llama.cpp,
/// MLX safetensors for MLX, `.task`/`.litertlm` for MediaPipe, `.pte` for
/// ExecuTorch, multi-`.mlmodelc` bundle for ANEMLL, single `.mlpackage` for
/// CoreML LLM. Each adapter publishes its own `supportedModels` list.
public struct ModelInfo: Codable, Sendable, Hashable {
    public let id: String
    public let displayName: String
    public let quantization: String
    public let parameterCountB: Double?
    public let onDiskSizeMB: Double?

    /// HuggingFace repo (`namespace/name`) the artifact lives in.
    public let hfRepoId: String
    /// Glob patterns to match when downloading. `["*"]` = whole snapshot.
    public let hfFilePatterns: [String]
    /// Path inside the downloaded snapshot, relative to the snapshot root.
    /// Empty string = the snapshot root itself.
    public let primaryFile: String

    public init(
        id: String,
        displayName: String,
        quantization: String,
        parameterCountB: Double? = nil,
        onDiskSizeMB: Double? = nil,
        hfRepoId: String,
        hfFilePatterns: [String] = ["*"],
        primaryFile: String = ""
    ) {
        self.id = id
        self.displayName = displayName
        self.quantization = quantization
        self.parameterCountB = parameterCountB
        self.onDiskSizeMB = onDiskSizeMB
        self.hfRepoId = hfRepoId
        self.hfFilePatterns = hfFilePatterns
        self.primaryFile = primaryFile
    }
}

/// Aggregate view of all models the app can run, grouped by runtime.
public enum ModelCatalog {
    /// Models the MLX Swift adapter can load. Curated 2026-05 — all
    /// entries verified present on `huggingface.co/mlx-community`.
    /// Sizes are approximate (verified via HF API or by download).
    public static let mlx: [ModelInfo] = [
        // --- Tiny — fits any device, including iPhone with headroom ---
        ModelInfo(
            id: "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            displayName: "Qwen 2.5 0.5B (4-bit)",
            quantization: "Q4",
            parameterCountB: 0.5,
            onDiskSizeMB: 300,
            hfRepoId: "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
        ),
        ModelInfo(
            id: "mlx-community/Qwen3-0.6B-4bit",
            displayName: "Qwen3-0.6B (4-bit)",
            quantization: "Q4",
            parameterCountB: 0.6,
            onDiskSizeMB: 350,
            hfRepoId: "mlx-community/Qwen3-0.6B-4bit"
        ),
        ModelInfo(
            id: "mlx-community/Qwen3-1.7B-4bit",
            displayName: "Qwen3-1.7B (4-bit)",
            quantization: "Q4",
            parameterCountB: 1.7,
            onDiskSizeMB: 1000,
            hfRepoId: "mlx-community/Qwen3-1.7B-4bit"
        ),
        ModelInfo(
            id: "mlx-community/Qwen3-4B-4bit",
            displayName: "Qwen3-4B (4-bit)",
            quantization: "Q4",
            parameterCountB: 4.0,
            onDiskSizeMB: 2300,
            hfRepoId: "mlx-community/Qwen3-4B-4bit"
        ),
        ModelInfo(
            id: "mlx-community/Qwen3-8B-4bit",
            displayName: "Qwen3-8B (4-bit)",
            quantization: "Q4",
            parameterCountB: 8.0,
            onDiskSizeMB: 4500,
            hfRepoId: "mlx-community/Qwen3-8B-4bit"
        ),
        ModelInfo(
            id: "mlx-community/Qwen3-14B-4bit",
            displayName: "Qwen3-14B (4-bit)",
            quantization: "Q4",
            parameterCountB: 14.0,
            onDiskSizeMB: 8000,
            hfRepoId: "mlx-community/Qwen3-14B-4bit"
        ),
        ModelInfo(
            id: "mlx-community/gemma-4-12b-it-4bit",
            displayName: "Gemma 4 12B (4-bit)",
            quantization: "Q4",
            parameterCountB: 12.0,
            onDiskSizeMB: 7000,
            hfRepoId: "mlx-community/gemma-4-12b-it-4bit"
        ),
        // Comparators for Lu's models. LFM2-350M is v2.0, kept for its published
        // rows; mlx-community has since published LFM2.5-1.2B (entry below,
        // 2026-08-27) — the size-matched mlx arm for the LFM2.5 litert cell.
        ModelInfo(
            id: "mlx-community/LFM2-350M-4bit",
            displayName: "LFM2-350M (4-bit)",
            quantization: "Q4",
            parameterCountB: 0.35,
            onDiskSizeMB: 200,
            hfRepoId: "mlx-community/LFM2-350M-4bit"
        ),
        ModelInfo(
            id: "mlx-community/LFM2.5-1.2B-Instruct-4bit",
            displayName: "LFM2.5-1.2B (4-bit)",
            quantization: "Q4",
            parameterCountB: 1.2,
            onDiskSizeMB: 700,
            hfRepoId: "mlx-community/LFM2.5-1.2B-Instruct-4bit"
        ),
        ModelInfo(
            id: "mlx-community/MiniCPM5-1B-4bit",
            displayName: "MiniCPM5-1B (4-bit)",
            quantization: "Q4",
            parameterCountB: 1.0,
            onDiskSizeMB: 600,
            hfRepoId: "mlx-community/MiniCPM5-1B-4bit"
        ),
        ModelInfo(
            id: "mlx-community/Qwen3.5-0.8B-MLX-4bit",
            displayName: "Qwen 3.5 0.8B (4-bit)",
            quantization: "Q4",
            parameterCountB: 0.8,
            onDiskSizeMB: 500,
            hfRepoId: "mlx-community/Qwen3.5-0.8B-MLX-4bit"
        ),
        // PTQ. Kept for the PTQ-vs-QAT delta; do NOT use it in a cross-runtime table — Google
        // ships a QAT build for every arm, so a PTQ row measures the recipe, not the runtime.
        // Measured on M4 Max: PTQ 78% vs QAT 87% on GSM8K (n=100, same protocol).
        ModelInfo(
            id: "mlx-community/gemma-4-e2b-it-4bit",
            displayName: "Gemma 4 E2B (PTQ 4-bit)",
            quantization: "INT4 (PTQ)",
            parameterCountB: 2.0,
            onDiskSizeMB: 1330,
            hfRepoId: "mlx-community/gemma-4-e2b-it-4bit"
        ),
        // The MLX arm's quality build. It sits beside the PTQ entry above because the two are
        // read against each other in every cross-runtime table, which only works if both can be
        // captured in one session — the published table has one row at 2026-07-27 and the other
        // at 07-19 precisely because this entry was missing from this checkout.
        ModelInfo(
            id: "mlx-community/gemma-4-e2b-it-qat-OptiQ-4bit",
            displayName: "Gemma 4 E2B (QAT 4-bit)",
            quantization: "INT4 (QAT, OptiQ)",
            parameterCountB: 2.0,
            onDiskSizeMB: 1330,
            hfRepoId: "mlx-community/gemma-4-e2b-it-qat-OptiQ-4bit"
        ),

        // --- Small — fits 16 GB Mac and iPhone Pro models ---
        ModelInfo(
            id: "mlx-community/Qwen3.5-2B-MLX-4bit",
            displayName: "Qwen 3.5 2B (4-bit)",
            quantization: "Q4",
            parameterCountB: 2.0,
            onDiskSizeMB: 1500,
            hfRepoId: "mlx-community/Qwen3.5-2B-MLX-4bit"
        ),
        ModelInfo(
            id: "mlx-community/gemma-4-e4b-it-4bit",
            displayName: "Gemma 4 E4B (4-bit)",
            quantization: "Q4",
            parameterCountB: 4.0,
            onDiskSizeMB: 3000,
            hfRepoId: "mlx-community/gemma-4-e4b-it-4bit"
        ),
        // QAT-int4 variant — the MLX arm's best build. NOTE: this does NOT give a "QAT-iso"
        // three-way with LiteRT; LiteRT runs the wNa8o8 mobile schema, which is a different
        // checkpoint that no other runtime can execute properly (see the .litertlm entry).
        // Each arm at its own best available quantization is the achievable comparison.
        ModelInfo(
            id: "mlx-community/gemma-4-e4b-it-qat-OptiQ-4bit",
            displayName: "Gemma 4 E4B (QAT 4-bit)",
            quantization: "INT4 (QAT, OptiQ)",
            parameterCountB: 4.0,
            onDiskSizeMB: 3000,
            hfRepoId: "mlx-community/gemma-4-e4b-it-qat-OptiQ-4bit"
        ),

        // --- Medium — Mac-class (M-series with ≥16 GB) ---
        ModelInfo(
            id: "mlx-community/Qwen3.5-9B-MLX-4bit",
            displayName: "Qwen 3.5 9B (4-bit)",
            quantization: "Q4",
            parameterCountB: 9.0,
            onDiskSizeMB: 5500,
            hfRepoId: "mlx-community/Qwen3.5-9B-MLX-4bit"
        ),

        // --- Large / MoE — workstation-class Mac ---
        ModelInfo(
            id: "mlx-community/gemma-4-26b-a4b-it-4bit",
            displayName: "Gemma 4 26B-A4B (4-bit, MoE)",
            quantization: "Q4",
            parameterCountB: 26.0,
            onDiskSizeMB: 14_500,
            hfRepoId: "mlx-community/gemma-4-26b-a4b-it-4bit"
        ),
        ModelInfo(
            id: "mlx-community/Qwen3.5-27B-4bit",
            displayName: "Qwen 3.5 27B (4-bit)",
            quantization: "Q4",
            parameterCountB: 27.0,
            onDiskSizeMB: 15_500,
            hfRepoId: "mlx-community/Qwen3.5-27B-4bit"
        ),
        ModelInfo(
            id: "mlx-community/Qwen3.5-35B-A3B-4bit",
            displayName: "Qwen 3.5 35B-A3B (4-bit, MoE)",
            quantization: "Q4",
            parameterCountB: 35.0,
            onDiskSizeMB: 19_500,
            hfRepoId: "mlx-community/Qwen3.5-35B-A3B-4bit"
        ),
        ModelInfo(
            id: "mlx-community/gemma-4-31b-it-4bit",
            displayName: "Gemma 4 31B (4-bit)",
            quantization: "Q4",
            parameterCountB: 31.0,
            onDiskSizeMB: 17_500,
            hfRepoId: "mlx-community/gemma-4-31b-it-4bit"
        ),

        // --- Cross-runtime comparison set (vs LiteRT-LM on iPhone 17 Pro) ---
        ModelInfo(
            id: "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit",
            displayName: "DeepSeek-R1-Distill-Qwen-1.5B (4-bit)",
            quantization: "Q4",
            parameterCountB: 1.5,
            onDiskSizeMB: 1000,
            hfRepoId: "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit"
        ),
        ModelInfo(
            id: "mlx-community/Phi-4-mini-instruct-4bit",
            displayName: "Phi-4-mini (4-bit)",
            quantization: "Q4",
            parameterCountB: 3.8,
            onDiskSizeMB: 2100,
            hfRepoId: "mlx-community/Phi-4-mini-instruct-4bit"
        ),
        ModelInfo(
            id: "mlx-community/gemma-3-1b-it-4bit",
            displayName: "Gemma3-1B-IT (4-bit)",
            quantization: "Q4",
            parameterCountB: 1.0,
            onDiskSizeMB: 700,
            hfRepoId: "mlx-community/gemma-3-1b-it-4bit"
        ),
        ModelInfo(
            id: "mlx-community/TinySwallow-1.5B-Instruct-4bit",
            displayName: "TinySwallow-1.5B (4-bit)",
            quantization: "Q4",
            parameterCountB: 1.5,
            onDiskSizeMB: 900,
            hfRepoId: "mlx-community/TinySwallow-1.5B-Instruct-4bit"
        ),
        ModelInfo(
            id: "mlx-community/Llama-3.2-3B-Instruct-4bit",
            displayName: "Llama-3.2-3B (4-bit)",
            quantization: "Q4",
            parameterCountB: 3.0,
            onDiskSizeMB: 1800,
            hfRepoId: "mlx-community/Llama-3.2-3B-Instruct-4bit"
        ),
        ModelInfo(
            id: "mlx-community/SmolLM3-3B-4bit",
            displayName: "SmolLM3-3B (4-bit)",
            quantization: "Q4",
            parameterCountB: 3.0,
            onDiskSizeMB: 1700,
            hfRepoId: "mlx-community/SmolLM3-3B-4bit"
        ),
        ModelInfo(
            id: "mlx-community/Ministral-3-3B-Instruct-2512-4bit",
            displayName: "Ministral-3-3B (4-bit)",
            quantization: "Q4",
            parameterCountB: 3.0,
            onDiskSizeMB: 1700,
            hfRepoId: "mlx-community/Ministral-3-3B-Instruct-2512-4bit"
        ),
    ]

    /// Models the llama.cpp adapter can load.
    /// One `.gguf` file per entry, downloaded from the listed HF repo.
    public static let llamaCpp: [ModelInfo] = [
        // Google's own QAT GGUF — the llama.cpp arm's best available build, and the one
        // cross-runtime tables should use. The third-party Q4_K_M below is PTQ.
        // ⚠️ UNLOADABLE as shipped (verified 2026-07-18): llama.cpp aborts loading its vocab —
        // "load: empty token at index 237922" then GGML_ASSERT(id_to_token.size() ==
        // token_to_id.size()) in llama-vocab.cpp. Reproduced on-device (vendored b8999) and
        // on macOS with 8680 AND the latest release b10064; the Q4_K_M below loads fine, so
        // it's this file's conversion, not gemma-4 support. Until Google re-exports or
        // llama.cpp tolerates the empty piece, the arm's official-QAT row reads "unloadable" —
        // which is itself the result. (Same lesson as wNa8o8: official artifact ≠ usable one.)
        ModelInfo(
            id: "google/gemma-4-E2B-it-qat-q4_0-gguf",
            displayName: "Gemma 4 E2B q4_0 QAT (GGUF)",
            quantization: "Q4_0 (QAT, official)",
            parameterCountB: 2.0,
            onDiskSizeMB: 3350,
            hfRepoId: "google/gemma-4-E2B-it-qat-q4_0-gguf",
            hfFilePatterns: ["gemma-4-E2B_q4_0-it.gguf"],
            primaryFile: "gemma-4-E2B_q4_0-it.gguf"
        ),
        // PTQ. Kept for the PTQ-vs-QAT delta; not for cross-runtime tables.
        ModelInfo(
            id: "unsloth/gemma-4-E2B-it-GGUF/Q4_K_M",
            displayName: "Gemma 4 E2B Q4_K_M (GGUF, PTQ)",
            quantization: "Q4_K_M (PTQ)",
            parameterCountB: 2.0,
            onDiskSizeMB: 1700,
            hfRepoId: "unsloth/gemma-4-E2B-it-GGUF",
            hfFilePatterns: ["gemma-4-E2B-it-Q4_K_M.gguf"],
            primaryFile: "gemma-4-E2B-it-Q4_K_M.gguf"
        ),
        ModelInfo(
            id: "unsloth/gemma-4-E4B-it-GGUF/Q4_K_M",
            displayName: "Gemma 4 E4B Q4_K_M (GGUF)",
            quantization: "Q4_K_M",
            parameterCountB: 4.0,
            onDiskSizeMB: 3300,
            hfRepoId: "unsloth/gemma-4-E4B-it-GGUF",
            hfFilePatterns: ["gemma-4-E4B-it-Q4_K_M.gguf"],
            primaryFile: "gemma-4-E4B-it-Q4_K_M.gguf"
        ),
        ModelInfo(
            id: "bartowski/Llama-3.2-1B-Instruct-GGUF/Q4_K_M",
            displayName: "Llama 3.2 1B Q4_K_M (GGUF)",
            quantization: "Q4_K_M",
            parameterCountB: 1.0,
            onDiskSizeMB: 800,
            hfRepoId: "bartowski/Llama-3.2-1B-Instruct-GGUF",
            hfFilePatterns: ["Llama-3.2-1B-Instruct-Q4_K_M.gguf"],
            primaryFile: "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
        ),
        ModelInfo(
            id: "bartowski/Qwen2.5-0.5B-Instruct-GGUF/Q4_K_M",
            displayName: "Qwen 2.5 0.5B Q4_K_M (GGUF)",
            quantization: "Q4_K_M",
            parameterCountB: 0.5,
            onDiskSizeMB: 380,
            hfRepoId: "bartowski/Qwen2.5-0.5B-Instruct-GGUF",
            hfFilePatterns: ["Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"],
            primaryFile: "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
        ),
        ModelInfo(
            id: "bartowski/Qwen_Qwen3.5-0.8B-GGUF/Q4_K_M",
            displayName: "Qwen 3.5 0.8B Q4_K_M (GGUF)",
            quantization: "Q4_K_M",
            parameterCountB: 0.8,
            onDiskSizeMB: 560,
            hfRepoId: "bartowski/Qwen_Qwen3.5-0.8B-GGUF",
            hfFilePatterns: ["Qwen_Qwen3.5-0.8B-Q4_K_M.gguf"],
            primaryFile: "Qwen_Qwen3.5-0.8B-Q4_K_M.gguf"
        ),
        ModelInfo(
            id: "unsloth/Qwen3.5-2B-GGUF/Q4_K_M",
            displayName: "Qwen 3.5 2B Q4_K_M (GGUF)",
            quantization: "Q4_K_M",
            parameterCountB: 2.0,
            onDiskSizeMB: 1300,
            hfRepoId: "unsloth/Qwen3.5-2B-GGUF",
            hfFilePatterns: ["Qwen3.5-2B-Q4_K_M.gguf"],
            primaryFile: "Qwen3.5-2B-Q4_K_M.gguf"
        ),
        ModelInfo(
            id: "unsloth/Qwen3.5-9B-GGUF/Q4_K_M",
            displayName: "Qwen 3.5 9B Q4_K_M (GGUF)",
            quantization: "Q4_K_M",
            parameterCountB: 9.0,
            onDiskSizeMB: 5500,
            hfRepoId: "unsloth/Qwen3.5-9B-GGUF",
            hfFilePatterns: ["Qwen3.5-9B-Q4_K_M.gguf"],
            primaryFile: "Qwen3.5-9B-Q4_K_M.gguf"
        ),
        ModelInfo(
            id: "unsloth/Qwen3-4B-GGUF/Q4_K_M",
            displayName: "Qwen3-4B Q4_K_M (GGUF)",
            quantization: "Q4_K_M",
            parameterCountB: 4.0,
            onDiskSizeMB: 2500,
            hfRepoId: "unsloth/Qwen3-4B-GGUF",
            hfFilePatterns: ["Qwen3-4B-Q4_K_M.gguf"],
            primaryFile: "Qwen3-4B-Q4_K_M.gguf"
        ),
        ModelInfo(
            id: "unsloth/Qwen3-8B-GGUF/Q4_K_M",
            displayName: "Qwen3-8B Q4_K_M (GGUF)",
            quantization: "Q4_K_M",
            parameterCountB: 8.0,
            onDiskSizeMB: 4900,
            hfRepoId: "unsloth/Qwen3-8B-GGUF",
            hfFilePatterns: ["Qwen3-8B-Q4_K_M.gguf"],
            primaryFile: "Qwen3-8B-Q4_K_M.gguf"
        ),
        ModelInfo(
            id: "unsloth/Qwen3-14B-GGUF/Q4_K_M",
            displayName: "Qwen3-14B Q4_K_M (GGUF)",
            quantization: "Q4_K_M",
            parameterCountB: 14.0,
            onDiskSizeMB: 9000,
            hfRepoId: "unsloth/Qwen3-14B-GGUF",
            hfFilePatterns: ["Qwen3-14B-Q4_K_M.gguf"],
            primaryFile: "Qwen3-14B-Q4_K_M.gguf"
        ),
        ModelInfo(
            id: "unsloth/gemma-4-12B-it-GGUF/Q4_K_M",
            displayName: "Gemma 4 12B Q4_K_M (GGUF)",
            quantization: "Q4_K_M",
            parameterCountB: 12.0,
            onDiskSizeMB: 7000,
            hfRepoId: "unsloth/gemma-4-12B-it-GGUF",
            hfFilePatterns: ["gemma-4-12b-it-Q4_K_M.gguf"],
            primaryFile: "gemma-4-12b-it-Q4_K_M.gguf"
        ),
        ModelInfo(
            id: "LiquidAI/LFM2.5-350M-GGUF/Q4_K_M",
            displayName: "LFM2.5-350M Q4_K_M (GGUF)",
            quantization: "Q4_K_M",
            parameterCountB: 0.35,
            onDiskSizeMB: 230,
            hfRepoId: "LiquidAI/LFM2.5-350M-GGUF",
            hfFilePatterns: ["LFM2.5-350M-Q4_K_M.gguf"],
            primaryFile: "LFM2.5-350M-Q4_K_M.gguf"
        ),
    ]

    /// Models the LiteRT-LM adapter can load.
    ///
    /// Wires `google-ai-edge/LiteRT-LM` ≥ 0.13 (the official Swift API,
    /// `import LiteRTLM`), which reads Google's `.litertlm` bundles. The
    /// catalog is **not** Gemma-only — `litert-community` ships Qwen3
    /// (0.6B/4B), LFM/Liquid, and others in `.litertlm` alongside Gemma; we
    /// target Qwen3-0.6B here so it lines up with the existing Qwen3-0.6B
    /// rows on MLX / CoreML / Core AI. This path supersedes the old MediaPipe
    /// 0.10.x (`.task`) path, which could not read Gemma 4 at all.
    ///
    /// Sizes are the standard (non-web, non-NPU) Metal-GPU variant. Gemma 4
    /// `.litertlm` is QAT INT4 on the decoder with the embedding table kept
    /// in higher precision and memory-mapped (E2B ≈ 0.79 GB decoder + 1.12 GB
    /// mmap'd embeddings ≈ 2.59 GB on disk). Context window 32k. Qwen3-0.6B is
    /// the mixed blockwise-INT4 artifact (gs32 weights, INT8 embeddings).
    public static let liteRTLM: [ModelInfo] = [
        // NOT uniform int4 — this is Google's wNa8o8 MOBILE schema (bit-identical to
        // `google/gemma-4-E2B-it-qat-mobile-transformers`): targeted 2-bit decode layers,
        // optimized KV cache, and STATIC INT8 ACTIVATIONS. It is a co-designed weights+runtime
        // package, not a bit-width, and it does not transfer: those same weights score 85% on
        // LiteRT and 48% on an fp16-activation runtime (GSM8K n=100, measured 2026-07-17 —
        // two independent fp16 implementations return identical wrong answers).
        // So its memory and decode wins here are partly the checkpoint's, not the runtime's;
        // no other arm can adopt this build to level the field.
        ModelInfo(
            id: "litert-community/gemma-4-E2B-it-litert-lm",
            displayName: "Gemma 4 E2B (.litertlm)",
            quantization: "wNa8o8 (int2/int4/int8 + int8 activations, QAT)",
            parameterCountB: 2.0,
            onDiskSizeMB: 2650,
            hfRepoId: "litert-community/gemma-4-E2B-it-litert-lm",
            hfFilePatterns: ["gemma-4-E2B-it.litertlm"],
            primaryFile: "gemma-4-E2B-it.litertlm"
        ),
        // Qwen3-0.6B — the model Lu's team is optimising; the 4-bit `.litertlm`
        // lines up with the existing Qwen3-0.6B rows on MLX / CoreML / Core AI.
        // Fallback if it won't load on GPU: the standard dynamic-INT8
        // `Qwen3-0.6B.litertlm` (614 MB) in the same repo.
        ModelInfo(
            id: "litert-community/Qwen3-0.6B",
            displayName: "Qwen3 0.6B (.litertlm)",
            quantization: "INT4 (mixed, blockwise gs32)",
            parameterCountB: 0.6,
            onDiskSizeMB: 498,
            hfRepoId: "litert-community/Qwen3-0.6B",
            hfFilePatterns: ["qwen3_0_6b_mixed_int4.litertlm"],
            primaryFile: "qwen3_0_6b_mixed_int4.litertlm"
        ),
        // Lu's focus models (Liquid/LFM2 + MiniCPM) — NOT on litert-community, so these are
        // OUR own .litertlm conversions, side-loaded (no HF download; hfRepoId is a local marker).
        // See MODEL_AVAILABILITY.md / MODEL_MATRIX.md.
        ModelInfo(
            id: "litert-local/lfm2.5-350m",
            displayName: "LFM2.5-350M (.litertlm, local)",
            quantization: "INT4 (ekv1024)",
            parameterCountB: 0.35,
            onDiskSizeMB: 178,
            hfRepoId: "litert-local/LFM2.5-350M",
            hfFilePatterns: ["LFM2.5-350M_int4_ekv1024.litertlm"],
            primaryFile: "LFM2.5-350M_int4_ekv1024.litertlm"
        ),
        // MiniCPM/LFM went PUBLIC on litert-community in 2026-08 (the "not on
        // litert-community" note above predates that) — prefer the published
        // artifacts; the local conversions stay only where no public equivalent
        // exists (LFM2.5-350M chat).
        ModelInfo(
            id: "litert-community/MiniCPM5-1B",
            displayName: "MiniCPM5-1B (.litertlm)",
            quantization: "wi4b32_wi8_afp32 (gpu-opt)",
            parameterCountB: 1.0,
            onDiskSizeMB: 793,
            hfRepoId: "litert-community/MiniCPM5-1B",
            hfFilePatterns: ["minicpm_wi4b32_wi8_afp32_gpu_opt.litertlm"],
            primaryFile: "minicpm_wi4b32_wi8_afp32_gpu_opt.litertlm"
        ),
        ModelInfo(
            id: "litert-community/LFM2.5-1.2B-Instruct",
            displayName: "LFM2.5-1.2B-Instruct (.litertlm)",
            quantization: "int4_gpu (litert-community descriptor)",
            parameterCountB: 1.2,
            onDiskSizeMB: 736,
            hfRepoId: "litert-community/LFM2.5-1.2B-Instruct",
            hfFilePatterns: ["LFM2.5-1.2B-Instruct_int4_gpu.litertlm"],
            primaryFile: "LFM2.5-1.2B-Instruct_int4_gpu.litertlm"
        ),
        // Qwen3-1.7B — litert-community has NO 1.7B (only 0.6B/4B/8B/14B), so these are
        // OUR own conversions (scripts/export_coreai_qwen3.sh has the Core AI side; the
        // LiteRT side is `litertlm-convert/export_simple_template.py`). Two LiteRT rows:
        //   • int8 (safe baseline, `dynamic_wi8_afp32`)
        //   • int4 mixed (`MIXED4`: int4 body + int8 tied-embedding/lm_head). The earlier
        //     "PTQ int4 collapses sub-2B" was specifically the EMBEDDING at int4 — keep it
        //     int8 and the int4 body is coherent (7/8 Mac, +28% decode vs int8). Both
        //     disclosed vs the official 0.6B/4B int4-QAT rows.
        // Side-loaded (no HF download); the 1.7B is the iPhone-ceiling probe between the
        // 0.6B that invokes and the 4B that invoke-fails on iOS.
        ModelInfo(
            id: "litert-local/qwen3-1.7b",
            displayName: "Qwen3-1.7B (.litertlm, local int8)",
            quantization: "INT8 (dynamic, ekv1024)",
            parameterCountB: 1.7,
            onDiskSizeMB: 1663,
            hfRepoId: "litert-local/Qwen3-1.7B",
            hfFilePatterns: ["*.litertlm"],
            primaryFile: "model.litertlm"
        ),
        ModelInfo(
            id: "litert-local/qwen3-1.7b-int4",
            displayName: "Qwen3-1.7B (.litertlm, local int4 mixed)",
            quantization: "INT4 (mixed, int8 embed)",
            parameterCountB: 1.7,
            onDiskSizeMB: 1438,
            hfRepoId: "litert-local/Qwen3-1.7B-int4",
            hfFilePatterns: ["*.litertlm"],
            primaryFile: "model.litertlm"
        ),
        // 2026-06-26 supplementary: byte-matched int4 of the litert-community q8 DeepSeek, to measure the
        // int8-vs-delegate split (keep the official q8 row; this is the "LiteRT-could-do-int4" companion).
        ModelInfo(
            id: "own/DeepSeek-R1-1.5B-int4-BOCTAV4",
            displayName: "DeepSeek-R1-1.5B (.litertlm, own int4 BOCTAV4)",
            quantization: "INT4 (BOCTAV4 blockwise-32 OCTAV, int8 embed)",
            parameterCountB: 1.5,
            onDiskSizeMB: 1024,
            hfRepoId: "own/DeepSeek-R1-1.5B-int4-BOCTAV4",
            hfFilePatterns: ["*.litertlm"],
            primaryFile: "model.litertlm"
        ),
        // int4 companions for the two remaining int8-shipped litert-community models
        // (Shuangfeng's all-int4 apple-to-apple ask, 2026-07-14). Same BOCTAV4 recipe as DeepSeek.
        ModelInfo(
            id: "own/TinySwallow-1.5B-int4-BOCTAV4",
            displayName: "TinySwallow-1.5B (.litertlm, own int4 BOCTAV4)",
            quantization: "INT4 (BOCTAV4 blockwise-32 OCTAV, int8 embed)",
            parameterCountB: 1.5,
            onDiskSizeMB: 1024,
            hfRepoId: "own/TinySwallow-1.5B-int4-BOCTAV4",
            hfFilePatterns: ["*.litertlm"],
            primaryFile: "model.litertlm"
        ),
        ModelInfo(
            id: "own/VibeThinker-1.5B-int4-BOCTAV4",
            displayName: "VibeThinker-1.5B (.litertlm, own int4 BOCTAV4)",
            quantization: "INT4 (BOCTAV4 blockwise-32 OCTAV, int8 embed)",
            parameterCountB: 1.5,
            onDiskSizeMB: 1024,
            hfRepoId: "own/VibeThinker-1.5B-int4-BOCTAV4",
            hfFilePatterns: ["*.litertlm"],
            primaryFile: "model.litertlm"
        ),
        ModelInfo(
            id: "own/Phi-4-mini-int4-BOCTAV4-128",
            displayName: "Phi-4-mini (.litertlm, own int4 BOCTAV4-128)",
            quantization: "INT4 (BOCTAV4 blockwise-128 OCTAV, int8 embed, static-rope)",
            parameterCountB: 3.8,
            onDiskSizeMB: 2200,
            hfRepoId: "own/Phi-4-mini-int4-BOCTAV4-128",
            hfFilePatterns: ["*.litertlm"],
            primaryFile: "model.litertlm"
        ),
        // Qwen3 4B / 8B — same mixed-INT4 .litertlm line as 0.6B, for a size-scaling
        // curve (0.6B → 4B → 8B). 8B (~4.4 GB) is desktop/Mac-tier; on phones it can
        // exceed the per-app memory ceiling (gemma-3n-style jetsam), so it stays Mac-only.
        ModelInfo(
            id: "litert-community/Qwen3-4B",
            displayName: "Qwen3 4B (.litertlm)",
            quantization: "INT4 (mixed, blockwise gs32)",
            parameterCountB: 4.0,
            onDiskSizeMB: 2300,
            hfRepoId: "litert-community/Qwen3-4B",
            hfFilePatterns: ["qwen3_4b_mixed_int4.litertlm"],
            primaryFile: "qwen3_4b_mixed_int4.litertlm"
        ),
        ModelInfo(
            id: "litert-community/Qwen3-8B",
            displayName: "Qwen3 8B (.litertlm)",
            quantization: "INT4 (mixed, blockwise gs32)",
            parameterCountB: 8.0,
            onDiskSizeMB: 4400,
            hfRepoId: "litert-community/Qwen3-8B",
            hfFilePatterns: ["qwen3_8b_mixed_int4.litertlm"],
            primaryFile: "qwen3_8b_mixed_int4.litertlm"
        ),
        ModelInfo(
            id: "litert-community/Qwen3-14B",
            displayName: "Qwen3 14B (.litertlm)",
            quantization: "INT4 (mixed, blockwise gs32)",
            parameterCountB: 14.0,
            onDiskSizeMB: 8000,
            hfRepoId: "litert-community/Qwen3-14B",
            hfFilePatterns: ["qwen3_14b_mixed_int4.litertlm"],
            primaryFile: "qwen3_14b_mixed_int4.litertlm"
        ),
        ModelInfo(
            id: "litert-community/gemma-4-12B-it-litert-lm",
            displayName: "Gemma 4 12B (.litertlm)",
            quantization: "INT4 (QAT)",
            parameterCountB: 12.0,
            onDiskSizeMB: 7000,
            hfRepoId: "litert-community/gemma-4-12B-it-litert-lm",
            hfFilePatterns: ["gemma-4-12B-it.litertlm"],
            primaryFile: "gemma-4-12B-it.litertlm"
        ),
        ModelInfo(
            id: "litert-community/gemma-4-E4B-it-litert-lm",
            displayName: "Gemma 4 E4B (.litertlm)",
            quantization: "INT4 (QAT)",
            parameterCountB: 4.0,
            onDiskSizeMB: 3750,
            hfRepoId: "litert-community/gemma-4-E4B-it-litert-lm",
            hfFilePatterns: ["gemma-4-E4B-it.litertlm"],
            primaryFile: "gemma-4-E4B-it.litertlm"
        ),
        ModelInfo(
            id: "google/gemma-3n-E2B-it-litert-lm",
            displayName: "Gemma 3n E2B (.litertlm)",
            quantization: "INT4 (QAT)",
            parameterCountB: 2.0,
            onDiskSizeMB: 3100,
            hfRepoId: "google/gemma-3n-E2B-it-litert-lm",
            hfFilePatterns: ["*.litertlm"],
            primaryFile: ""
        ),

        // --- Cross-runtime comparison set (vs MLX on iPhone 17 Pro) ---
        // litert-community DOWNLOAD entries (fetched on-device).
        ModelInfo(
            id: "litert-community/Gemma3-1B-IT",
            displayName: "Gemma3-1B (.litertlm, official int4)",
            quantization: "INT4",
            parameterCountB: 1.0,
            onDiskSizeMB: 700,
            hfRepoId: "litert-community/Gemma3-1B-IT",
            hfFilePatterns: ["gemma3-1b-it-int4.litertlm"],
            primaryFile: "gemma3-1b-it-int4.litertlm"
        ),
        ModelInfo(
            id: "litert-community/Phi-4-mini-instruct",
            displayName: "Phi-4-mini (.litertlm)",
            quantization: "INT8",
            parameterCountB: 3.8,
            onDiskSizeMB: 3600,
            hfRepoId: "litert-community/Phi-4-mini-instruct",
            hfFilePatterns: ["Phi-4-mini-instruct_multi-prefill-seq_q8_ekv4096.litertlm"],
            primaryFile: "Phi-4-mini-instruct_multi-prefill-seq_q8_ekv4096.litertlm"
        ),
        ModelInfo(
            id: "litert-community/DeepSeek-R1-Distill-Qwen-1.5B",
            displayName: "DeepSeek-R1-Distill-Qwen-1.5B (.litertlm)",
            quantization: "INT8",
            parameterCountB: 1.5,
            onDiskSizeMB: 1700,
            hfRepoId: "litert-community/DeepSeek-R1-Distill-Qwen-1.5B",
            hfFilePatterns: ["DeepSeek-R1-Distill-Qwen-1.5B_multi-prefill-seq_q8_ekv4096.litertlm"],
            primaryFile: "DeepSeek-R1-Distill-Qwen-1.5B_multi-prefill-seq_q8_ekv4096.litertlm"
        ),
        ModelInfo(
            id: "litert-community/TinySwallow-1.5B-Instruct",
            displayName: "TinySwallow-1.5B (.litertlm)",
            quantization: "INT8",
            parameterCountB: 1.5,
            onDiskSizeMB: 1700,
            hfRepoId: "litert-community/TinySwallow-1.5B-Instruct",
            hfFilePatterns: ["TinySwallow-1.5B-Instruct.litertlm"],
            primaryFile: "TinySwallow-1.5B-Instruct.litertlm"
        ),
        ModelInfo(
            id: "litert-community/VibeThinker-1.5B",
            displayName: "VibeThinker-1.5B (.litertlm)",
            quantization: "INT8",
            parameterCountB: 1.5,
            onDiskSizeMB: 1700,
            hfRepoId: "litert-community/VibeThinker-1.5B",
            hfFilePatterns: ["VibeThinker-1.5B.litertlm"],
            primaryFile: "VibeThinker-1.5B.litertlm"
        ),
        // litert-local SIDE-LOAD entries (placed on device at
        // Documents/models/litert-lm/<hfRepoId "/"→"__">/model.litertlm).
        ModelInfo(
            id: "litert-local/olmo2-1b",
            displayName: "OLMo-2-1B (.litertlm, local int4)",
            quantization: "INT4",
            parameterCountB: 1.0,
            onDiskSizeMB: 888,
            hfRepoId: "litert-local/OLMo-2-1B",
            hfFilePatterns: ["*.litertlm"],
            primaryFile: "model.litertlm"
        ),
        ModelInfo(
            id: "litert-local/llama32-3b",
            displayName: "Llama-3.2-3B (.litertlm, local int4)",
            quantization: "INT4",
            parameterCountB: 3.0,
            onDiskSizeMB: 2100,
            hfRepoId: "litert-local/Llama-3.2-3B",
            hfFilePatterns: ["*.litertlm"],
            primaryFile: "model.litertlm"
        ),
        ModelInfo(
            id: "litert-local/smollm3-3b",
            displayName: "SmolLM3-3B (.litertlm, local int4)",
            quantization: "INT4",
            parameterCountB: 3.0,
            onDiskSizeMB: 1900,
            hfRepoId: "litert-local/SmolLM3-3B",
            hfFilePatterns: ["*.litertlm"],
            primaryFile: "model.litertlm"
        ),
        ModelInfo(
            id: "litert-local/ministral3-3b",
            displayName: "Ministral-3-3B (.litertlm, local int4)",
            quantization: "INT4",
            parameterCountB: 3.0,
            onDiskSizeMB: 2200,
            hfRepoId: "litert-local/Ministral-3-3B",
            hfFilePatterns: ["*.litertlm"],
            primaryFile: "model.litertlm"
        ),
    ]

    /// Back-compat alias. The adapter and earlier call sites refer to this
    /// runtime's catalog as `mediaPipe`; LiteRT-LM is the current name.
    public static let mediaPipe: [ModelInfo] = liteRTLM

    /// Models the ExecuTorch adapter can load.
    /// Each repo ships a `.pte` plus a `tokenizer.model` (sentencepiece).
    /// No official Gemma 4 .pte exists yet (May 2026).
    public static let executorch: [ModelInfo] = [
        ModelInfo(
            id: "executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET",
            displayName: "Llama 3.2 1B SpinQuant INT4 (.pte)",
            quantization: "INT4 (SpinQuant)",
            parameterCountB: 1.0,
            onDiskSizeMB: 1140,
            hfRepoId: "executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET",
            hfFilePatterns: ["*.pte", "tokenizer.model"],
            primaryFile: "Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte"
        ),
    ]

    /// Models the ANEMLL adapter can load.
    /// Each repo ships a multi-file bundle (embeddings + FFN chunks + lm_head + meta.yaml + tokenizer files).
    public static let anemll: [ModelInfo] = [
        ModelInfo(
            id: "anemll/anemll-google-gemma-3-1b-it-ctx4096_0.3.5",
            displayName: "Gemma 3 1B IT (ANEMLL ANE)",
            quantization: "Q4 (ANE-tuned)",
            parameterCountB: 1.0,
            onDiskSizeMB: 1400,
            hfRepoId: "anemll/anemll-google-gemma-3-1b-it-ctx4096_0.3.5"
        ),
        ModelInfo(
            id: "anemll/anemll-meta-llama-Llama-3.2-1B-Instruct-ctx1024_0.3.5",
            displayName: "Llama 3.2 1B Instruct (ANEMLL ANE)",
            quantization: "Q4 (ANE-tuned)",
            parameterCountB: 1.0,
            onDiskSizeMB: 1500,
            hfRepoId: "anemll/anemll-meta-llama-Llama-3.2-1B-Instruct-ctx1024_0.3.5"
        ),
        ModelInfo(
            id: "anemll/anemll-google-gemma-3-270m-it-ctx4096_0.3.5",
            displayName: "Gemma 3 270M IT (ANEMLL ANE)",
            quantization: "Q4 (ANE-tuned)",
            parameterCountB: 0.27,
            onDiskSizeMB: 400,
            hfRepoId: "anemll/anemll-google-gemma-3-270m-it-ctx4096_0.3.5"
        ),
    ]

    /// Models the CoreML LLM adapter (john-rocky/CoreML-LLM) can load.
    /// Each id maps to a registered `CoreMLLLM.ModelDownloader.ModelInfo`
    /// that downloads and ANE-compiles a chunked `.mlmodelc` bundle from
    /// the `mlboydaisuke/*-coreml` HF namespace on first use.
    public static let coreML: [ModelInfo] = [
        ModelInfo(
            id: "coreml-llm/gemma4-e2b",
            displayName: "Gemma 4 E2B (CoreML, ANE)",
            quantization: "INT4 palettized",
            parameterCountB: 2.0,
            onDiskSizeMB: 5400,
            hfRepoId: "mlboydaisuke/gemma-4-E2B-coreml"
        ),
        ModelInfo(
            id: "coreml-llm/gemma4-e4b",
            displayName: "Gemma 4 E4B (CoreML, ANE)",
            quantization: "INT4 palettized",
            parameterCountB: 4.0,
            onDiskSizeMB: 5500,
            hfRepoId: "mlboydaisuke/gemma-4-E4B-coreml"
        ),
        ModelInfo(
            id: "coreml-llm/qwen3.5-0.8b",
            displayName: "Qwen 3.5 0.8B (CoreML, ANE)",
            quantization: "INT8",
            parameterCountB: 0.8,
            onDiskSizeMB: 1200,
            hfRepoId: "mlboydaisuke/qwen3.5-0.8B-CoreML"
        ),
        ModelInfo(
            id: "coreml-llm/qwen3.5-2b",
            displayName: "Qwen 3.5 2B (CoreML, ANE)",
            quantization: "INT8",
            parameterCountB: 2.0,
            onDiskSizeMB: 2800,
            hfRepoId: "mlboydaisuke/qwen3.5-2B-CoreML"
        ),
        ModelInfo(
            id: "coreml-llm/lfm2.5-350m",
            displayName: "LFM 2.5 350M (CoreML, ANE)",
            quantization: "INT8",
            parameterCountB: 0.35,
            onDiskSizeMB: 810,
            hfRepoId: "mlboydaisuke/lfm2.5-350m-coreml"
        ),
        ModelInfo(
            id: "coreml-llm/qwen2.5-0.5b",
            displayName: "Qwen 2.5 0.5B (CoreML, text)",
            quantization: "FP16",
            parameterCountB: 0.5,
            onDiskSizeMB: 309,
            hfRepoId: "mlboydaisuke/qwen2.5-0.5b-coreml"
        ),
        ModelInfo(
            id: "coreml-llm/qwen3-0.6b",
            displayName: "Qwen3-0.6B (CoreML, ANE)",
            quantization: "INT8 palettized",
            parameterCountB: 0.6,
            onDiskSizeMB: 900,
            hfRepoId: "mlboydaisuke/qwen3-0.6b-coreml"
        ),
    ]

    /// Models the Apple Foundation Models adapter can run.
    ///
    /// Apple FM is a single, pre-installed on-device model — no HF download,
    /// no model picking. The catalog still carries one entry so the harness
    /// can attach a stable id / display name to the runs. Size / quant are
    /// best-guess (Apple has not published a parameter count for the GA
    /// model; community estimates put it at ~3B params with 2-bit / 4-bit
    /// adapters).
    public static let appleFM: [ModelInfo] = [
        ModelInfo(
            id: "apple-fm/default",
            displayName: "Apple Foundation Model (default, on-device)",
            quantization: "Apple-quant (~2-4 bit, adapters)",
            parameterCountB: 3.0,
            onDiskSizeMB: nil,
            hfRepoId: ""
        ),
    ]

    /// Models the Apple **Core AI** adapter can run — the same Qwen3-0.6B
    /// `.aimodel` bundle exported by the official `coreai.llm.export` pipeline,
    /// exposed twice so the harness can benchmark both compute paths:
    /// the Neural Engine (`static-shape`) and the GPU (`coreai-pipelined`).
    ///
    /// The bundle is side-loaded into `Documents/CoreAIModels/qwen3_0_6b_ios/`
    /// (the `.aimodel` is ~434 MB and not published to HF), so `hfRepoId` is
    /// empty. Requires iOS 27 + the `coreai-models` Swift package.
    public static let coreAI: [ModelInfo] = [
        ModelInfo(
            id: "core-ai/qwen3-0.6b-ane",
            displayName: "Qwen3-0.6B (Core AI, ANE)",
            quantization: "4-bit palettized (uniform g32)",
            parameterCountB: 0.6,
            onDiskSizeMB: 389,
            hfRepoId: ""
        ),
        ModelInfo(
            id: "core-ai/qwen3-0.6b-ane-june",
            displayName: "Qwen3-0.6B (Core AI, ANE, June mixed-4/8 lineage)",
            quantization: "mixed 4/8-bit (June static export)",
            parameterCountB: 0.6,
            onDiskSizeMB: 732,
            hfRepoId: ""
        ),
        ModelInfo(
            id: "core-ai/qwen3-1.7b-gpu-june",
            displayName: "Qwen3-1.7B (Core AI, GPU, June lineage)",
            quantization: "INT4 (dynamic, June export)",
            parameterCountB: 1.7,
            onDiskSizeMB: 940,
            hfRepoId: ""
        ),
        ModelInfo(
            id: "core-ai/qwen3-0.6b-gpu",
            displayName: "Qwen3-0.6B (Core AI, GPU)",
            // macOS-26-era official export (native quantized-Linear lowering) — the
            // 27-beta re-export of the same recipe is ~2.2x slower per the card;
            // the era is part of the recipe here.
            quantization: "INT4 (dynamic, macOS-26-era export)",
            parameterCountB: 0.6,
            onDiskSizeMB: 327,
            hfRepoId: ""
        ),
        ModelInfo(
            id: "core-ai/qwen3-1.7b-ane",
            displayName: "Qwen3-1.7B (Core AI, ANE)",
            quantization: "4-bit palettized (uniform g32)",
            parameterCountB: 1.7,
            onDiskSizeMB: 1400,
            hfRepoId: ""
        ),
        ModelInfo(
            id: "core-ai/qwen3-1.7b-gpu",
            displayName: "Qwen3-1.7B (Core AI, GPU)",
            quantization: "INT4 (dynamic)",
            parameterCountB: 1.7,
            onDiskSizeMB: 940,
            hfRepoId: ""
        ),
        ModelInfo(
            id: "core-ai/deepseek-r1-1.5b-ane",
            displayName: "DeepSeek-R1-1.5B (Core AI, ANE)",
            quantization: "4-bit palettized (uniform g32)",
            parameterCountB: 1.5, onDiskSizeMB: 1300, hfRepoId: ""
        ),
        ModelInfo(
            id: "core-ai/deepseek-r1-1.5b-gpu",
            displayName: "DeepSeek-R1-1.5B (Core AI, GPU)",
            quantization: "INT4 (dynamic)",
            parameterCountB: 1.5, onDiskSizeMB: 900, hfRepoId: ""
        ),
        // 2026-08-26: b2-refreshed canonical bundles (mlboydaisuke, 08-25) compiled
        // h18p/GPU for the cross-runtime demo pairs — staging: ~/bench-coreai-staging.
        ModelInfo(
            id: "core-ai/lfm2.5-1.2b-gpu",
            displayName: "LFM2.5-1.2B (Core AI, GPU)",
            quantization: "int8hu block32 sym (untied head)",
            parameterCountB: 1.2, onDiskSizeMB: 1624, hfRepoId: ""
        ),
        ModelInfo(
            id: "core-ai/minicpm5-1b-gpu",
            displayName: "MiniCPM5-1B (Core AI, GPU)",
            quantization: "INT8 (sym, dynamic)",
            parameterCountB: 1.0, onDiskSizeMB: 1042, hfRepoId: ""
        ),
        ModelInfo(
            id: "core-ai/tinyswallow-1.5b-ane",
            displayName: "TinySwallow-1.5B (Core AI, ANE)",
            quantization: "4-bit palettized (uniform g32)",
            parameterCountB: 1.5, onDiskSizeMB: 1300, hfRepoId: ""
        ),
        ModelInfo(
            id: "core-ai/tinyswallow-1.5b-gpu",
            displayName: "TinySwallow-1.5B (Core AI, GPU)",
            quantization: "INT4 (dynamic)",
            parameterCountB: 1.5, onDiskSizeMB: 900, hfRepoId: ""
        ),
        ModelInfo(
            id: "core-ai/vibethinker-1.5b-ane",
            displayName: "VibeThinker-1.5B (Core AI, ANE)",
            quantization: "4-bit palettized (uniform g32)",
            parameterCountB: 1.5, onDiskSizeMB: 1300, hfRepoId: ""
        ),
        ModelInfo(
            id: "core-ai/vibethinker-1.5b-gpu",
            displayName: "VibeThinker-1.5B (Core AI, GPU)",
            quantization: "INT4 (dynamic)",
            parameterCountB: 1.5, onDiskSizeMB: 900, hfRepoId: ""
        ),
        // 2026-06-25 export pass — GPU for all 6; ANE only llama/olmo2/smollm3 (ministral/gemma3/phi ANE pending)
        ModelInfo(
            id: "core-ai/ministral-3b-gpu",
            displayName: "Ministral-3-3B (Core AI, GPU)",
            quantization: "INT4 (dynamic)",
            parameterCountB: 3.0, onDiskSizeMB: 1800, hfRepoId: ""
        ),
        ModelInfo(
            id: "core-ai/gemma3-1b-gpu",
            displayName: "Gemma3-1B (Core AI, GPU)",
            quantization: "INT4 (dynamic)",
            parameterCountB: 1.0, onDiskSizeMB: 574, hfRepoId: ""
        ),
        ModelInfo(
            id: "core-ai/phi-4-mini-gpu",
            displayName: "Phi-4-mini (Core AI, GPU)",
            quantization: "INT4 (dynamic)",
            parameterCountB: 3.8, onDiskSizeMB: 2000, hfRepoId: ""
        ),
        ModelInfo(
            id: "core-ai/llama-3.2-3b-ane",
            displayName: "Llama-3.2-3B (Core AI, ANE)",
            quantization: "4-bit palettized (uniform g32)",
            parameterCountB: 3.0, onDiskSizeMB: 2300, hfRepoId: ""
        ),
        ModelInfo(
            id: "core-ai/llama-3.2-3b-gpu",
            displayName: "Llama-3.2-3B (Core AI, GPU)",
            quantization: "INT4 (dynamic)",
            parameterCountB: 3.0, onDiskSizeMB: 1700, hfRepoId: ""
        ),
        ModelInfo(
            id: "core-ai/olmo2-1b-ane",
            displayName: "OLMo-2-1B (Core AI, ANE)",
            quantization: "4-bit palettized (uniform g32)",
            parameterCountB: 1.0, onDiskSizeMB: 1100, hfRepoId: ""
        ),
        ModelInfo(
            id: "core-ai/olmo2-1b-gpu",
            displayName: "OLMo-2-1B (Core AI, GPU)",
            quantization: "INT4 (dynamic)",
            parameterCountB: 1.0, onDiskSizeMB: 806, hfRepoId: ""
        ),
        ModelInfo(
            id: "core-ai/smollm3-3b-ane",
            displayName: "SmolLM3-3B (Core AI, ANE)",
            quantization: "4-bit palettized (uniform g32)",
            parameterCountB: 3.0, onDiskSizeMB: 2300, hfRepoId: ""
        ),
        ModelInfo(
            id: "core-ai/smollm3-3b-gpu",
            displayName: "SmolLM3-3B (Core AI, GPU)",
            quantization: "INT4 (dynamic)",
            parameterCountB: 3.0, onDiskSizeMB: 1600, hfRepoId: ""
        ),
        ModelInfo(
            id: "core-ai/qwen3-4b-ane",
            displayName: "Qwen3-4B (Core AI, ANE)",
            quantization: "4-bit palettized (uniform g32)",
            parameterCountB: 4.0,
            onDiskSizeMB: 2954,
            hfRepoId: ""
        ),
        ModelInfo(
            id: "core-ai/qwen3-4b-gpu",
            displayName: "Qwen3-4B (Core AI, GPU)",
            quantization: "INT4 (dynamic)",
            parameterCountB: 4.0,
            onDiskSizeMB: 2159,
            hfRepoId: ""
        ),
        ModelInfo(
            id: "core-ai/qwen3-8b-ane",
            displayName: "Qwen3-8B (Core AI, ANE)",
            quantization: "4-bit palettized (uniform g32)",
            parameterCountB: 8.0,
            onDiskSizeMB: 5317,
            hfRepoId: ""
        ),
        ModelInfo(
            id: "core-ai/qwen3-8b-gpu",
            displayName: "Qwen3-8B (Core AI, GPU)",
            quantization: "INT4 (dynamic)",
            parameterCountB: 8.0,
            onDiskSizeMB: 4396,
            hfRepoId: ""
        ),
        // 2026-06-26 static-GPU experiment: static-shape palettized bundle placed on GPU (0 ANE regions).
        // 3-way with *_ane (static/ANE) + *_gpu (dynamic/GPU). Side-load to Documents/CoreAIModels/<name>_static_gpu/.
        ModelInfo(id: "core-ai/deepseek-r1-1.5b-static-gpu", displayName: "DeepSeek-R1-1.5B (Core AI, static-GPU)", quantization: "4-bit palettized (uniform g32, static→GPU)", parameterCountB: 1.5, onDiskSizeMB: 992, hfRepoId: ""),
        ModelInfo(id: "core-ai/tinyswallow-1.5b-static-gpu", displayName: "TinySwallow-1.5B (Core AI, static-GPU)", quantization: "4-bit palettized (uniform g32, static→GPU)", parameterCountB: 1.5, onDiskSizeMB: 881, hfRepoId: ""),
        ModelInfo(id: "core-ai/vibethinker-1.5b-static-gpu", displayName: "VibeThinker-1.5B (Core AI, static-GPU)", quantization: "4-bit palettized (uniform g32, static→GPU)", parameterCountB: 1.5, onDiskSizeMB: 881, hfRepoId: ""),
        ModelInfo(id: "core-ai/qwen3-0.6b-static-gpu", displayName: "Qwen3-0.6B (Core AI, static-GPU)", quantization: "4-bit palettized (uniform g32, static→GPU)", parameterCountB: 0.6, onDiskSizeMB: 393, hfRepoId: ""),
        ModelInfo(id: "core-ai/qwen3-1.7b-static-gpu", displayName: "Qwen3-1.7B (Core AI, static-GPU)", quantization: "4-bit palettized (uniform g32, static→GPU)", parameterCountB: 1.7, onDiskSizeMB: 1000, hfRepoId: ""),
        ModelInfo(id: "core-ai/qwen3-4b-static-gpu", displayName: "Qwen3-4B (Core AI, static-GPU)", quantization: "4-bit palettized (uniform g32, static→GPU)", parameterCountB: 4.0, onDiskSizeMB: 2100, hfRepoId: ""),
        ModelInfo(id: "core-ai/qwen3-8b-static-gpu", displayName: "Qwen3-8B (Core AI, static-GPU)", quantization: "4-bit palettized (uniform g32, static→GPU)", parameterCountB: 8.0, onDiskSizeMB: 4200, hfRepoId: ""),
        ModelInfo(id: "core-ai/olmo2-1b-static-gpu", displayName: "OLMo-2-1B (Core AI, static-GPU)", quantization: "4-bit palettized (uniform g32, static→GPU)", parameterCountB: 1.0, onDiskSizeMB: 828, hfRepoId: ""),
        ModelInfo(id: "core-ai/smollm3-3b-static-gpu", displayName: "SmolLM3-3B (Core AI, static-GPU)", quantization: "4-bit palettized (uniform g32, static→GPU)", parameterCountB: 3.0, onDiskSizeMB: 1600, hfRepoId: ""),
        ModelInfo(id: "core-ai/llama-3.2-3b-static-gpu", displayName: "Llama-3.2-3B (Core AI, static-GPU)", quantization: "4-bit palettized (uniform g32, static→GPU)", parameterCountB: 3.0, onDiskSizeMB: 1700, hfRepoId: ""),
        // Gemma-4 E4B (Per-Layer-Embeddings): `_tbl` GPU-pipelined decode + mmap'd PLE table (in-graph gather).
        ModelInfo(id: "core-ai/gemma4-e4b-gpu", displayName: "Gemma 4 E4B (Core AI, GPU)", quantization: "int4 q4_0 (QAT)", parameterCountB: 4.0, onDiskSizeMB: 5300, hfRepoId: "mlboydaisuke/gemma-4-E4B-CoreAI"),
        // Gemma-4 E2B, same PLE structure (own export from google's -qat-q4_0-unquantized; Apple
        // ships no Gemma-4 bundle). PLE arm ⇒ needs the patched engine (COREAI_STATIC_INPUTS) —
        // a stock build reports it unsupported; any published number must be labelled
        // "patched engine (reference)". See methodology/core-ai-arm-provenance.md.
        ModelInfo(id: "core-ai/gemma4-e2b-gpu", displayName: "Gemma 4 E2B (Core AI, GPU)", quantization: "int4 q4_0 (QAT, own export)", parameterCountB: 2.0, onDiskSizeMB: 2048, hfRepoId: "mlboydaisuke/gemma-4-E2B-CoreAI"),
    ]

    /// Cactus (`cactus-compute/cactus`) — CQ bundles from `huggingface.co/Cactus-Compute`.
    /// Each bundle is a zip of a directory (graph + manifest + `config.txt`); the CLI's
    /// default fetch for `google/gemma-4-E2B-it` is the **CQ4** bundle, so that is the
    /// deployable-default row. The zips are not per-file downloadable through the HF
    /// snapshot API in a useful way, so provisioning is sideload-first: unzip the bundle
    /// and push the directory to `Documents/models/cactus/<repo>/<bundle-dir>/`
    /// (`HFDownloader.snapshot` short-circuits on a non-empty dir).
    public static let cactus: [ModelInfo] = [
        // The bundle their own `cactus benchmark`/`cactus run` downloads by default
        // (bits=4). Since 2026-07-09 this file is the "locally-built calibrated CQ4
        // (GPTQ language, FP16 towers)" (their commit title); 07-17 added a
        // cloud-handoff probe (probe-only change vs those graphs). CQ = Hadamard
        // rotation + per-group codebook PTQ.
        ModelInfo(
            id: "Cactus-Compute/gemma-4-E2B-it-cq4",
            displayName: "Gemma 4 E2B (Cactus CQ4)",
            quantization: "CQ4 (rotation+codebook PTQ, calibrated)",
            parameterCountB: 2.0,
            onDiskSizeMB: 3800,
            hfRepoId: "Cactus-Compute/gemma-4-E2B-it",
            hfFilePatterns: ["config.json"],
            primaryFile: "gemma-4-e2b-it-cq4"
        ),
        // The pre-2026-07-09 CQ4 bundle (renamed `-uncalibrated` when the "calibrated"
        // build replaced it as the default). Same repo, same engine, same format —
        // but it RETAINS multi-step reasoning where the shipped default does not
        // (GSM8K n=100 one-harness: uncalibrated 87.0% vs calibrated-default 3.0%,
        // measured 2026-07-20). Not CLI-fetchable (`-cq<bits>$` filename rule), so
        // provisioning is manual download + sideload; loads via plain `cactus_init`.
        ModelInfo(
            id: "Cactus-Compute/gemma-4-E2B-it-cq4-uncalibrated",
            displayName: "Gemma 4 E2B (Cactus CQ4, uncalibrated)",
            quantization: "CQ4 (rotation+codebook PTQ, uncalibrated pre-07-09 build)",
            parameterCountB: 2.0,
            onDiskSizeMB: 3800,
            hfRepoId: "Cactus-Compute/gemma-4-E2B-it",
            hfFilePatterns: ["config.json"],
            primaryFile: "gemma-4-e2b-it-cq4-uncalibrated"
        ),
    ]

    /// Default model picked when the app first launches.
    public static let defaultModel: ModelInfo = mlx[0]
}
