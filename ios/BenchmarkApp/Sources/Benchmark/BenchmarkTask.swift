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

    /// When non-nil, the runner repeats generation until this many seconds of
    /// active decode have elapsed, instead of running the prompt once. Used by
    /// the energy task so a measurable battery delta builds up (a single short
    /// reply is far below the iOS battery API's 1% step). `nil` = run once.
    var sustainSeconds: TimeInterval? { get }
}

public extension BenchmarkTask {
    var sustainSeconds: TimeInterval? { nil }
}

public enum BenchmarkTaskCatalog {
    public static let all: [any BenchmarkTask] = [
        ShortChatTask(),
        LongContextTask(id: "long-context-512", targetTokens: 512),     // context-length sweep within the 4096 ctx ceiling
        LongContextTask(),                                              // ~2K
        LongContextTask(id: "long-context-3k", targetTokens: 3072),     // near the 4096 ctx ceiling (room for 128 decode)
        LongContextTask(id: "long-context-8k", targetTokens: 8192),
        LongContextTask(id: "long-context-32k", targetTokens: 32768),
        CactusParityTask(),
        SustainedGenerationTask(),
        EnergyTask(),
        QualityTask(),
        AppLifecycleTask(),
    ]

    public static func task(for id: String) -> (any BenchmarkTask)? {
        all.first(where: { $0.id == id })
    }
}
