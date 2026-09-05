import Foundation

public protocol BenchmarkTask: Sendable {
    /// Stable id used for filenames and the result table.
    var id: String { get }

    /// Human-friendly title shown in the UI.
    var title: String { get }

    /// One-sentence description of what this task measures.
    var summary: String { get }

    /// The prompt to feed into the runtime.
    var prompt: String { get }

    /// Generation parameters specific to this task.
    var parameters: GenerationParameters { get }

    /// Name of an image resource bundled with the app (without extension). When
    /// non-nil the runner drives the vision path, passing the image alongside the
    /// prompt. `nil` (the default) = text-only task.
    var imageResource: String? { get }

    /// When non-nil, the runner repeats generation until this many seconds of
    /// active decode have elapsed, instead of running the prompt once. Used by
    /// the energy task so a measurable battery delta builds up (a single short
    /// reply is far below the iOS battery API's 1% step). `nil` = run once.
    var sustainSeconds: TimeInterval? { get }
}

public extension BenchmarkTask {
    var sustainSeconds: TimeInterval? { nil }
    var imageResource: String? { nil }
}

public enum BenchmarkTaskCatalog {
    public static let all: [any BenchmarkTask] = [
        ShortChatTask(),
        LongContextTask(id: "long-context-512", targetTokens: 512),     // context-length sweep within the 4096 ctx ceiling
        LongContextTask(id: "long-context-1024", targetTokens: 1024, maxTokens: 256),  // p=1024/g=256, one-sentence tail (the 7/18 deep-context cells)
        // Same p=1024 prefill, but the tail forces the model to fill the 256-token budget.
        // This is the only cross-arm instrument for the deep-context column: LiteRT-LM's
        // native benchmark() forces prefill without a prompt, and no other runtime has an
        // equivalent entry point, so the four remaining arms can only be measured here.
        LongContextTask(id: "long-context-1024-gen256", targetTokens: 1024, maxTokens: 256,
                        forceLongOutput: true),
        LongContextTask(),                                              // ~2K
        LongContextTask(id: "long-context-3k", targetTokens: 3072),     // near the 4096 ctx ceiling (room for 128 decode)
        LongContextTask(id: "long-context-8k", targetTokens: 8192),
        LongContextTask(id: "long-context-32k", targetTokens: 32768),
        CactusParityTask(),
        SustainedGenerationTask(),
        EnergyTask(),
        QualityTask(),
        // Single-token safety classifiers (Shieldstral-class): the verdict IS the
        // output, so these two replace the prose-quality task for that model family.
        SafetyClassifyTask.unsafeCase(),
        SafetyClassifyTask.safeCase(),
        SafetyClassifyTask.visionText(),
        SafetyClassifyTask.visionPlain(),
        AppLifecycleTask(),
    ]

    public static func task(for id: String) -> (any BenchmarkTask)? {
        all.first(where: { $0.id == id })
    }
}
