import Foundation

public enum RuntimeKind: String, CaseIterable, Codable, Sendable, Identifiable {
    case mlxSwift = "mlx-swift"
    case llamaCpp = "llama.cpp"
    case coreMLLLM = "coreml-llm"
    case mediaPipe = "litert-lm"
    case executorch = "executorch"
    case anemll = "anemll"
    case appleFM = "apple-fm"
    case coreAI = "core-ai"
    case cactus = "cactus"

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .mlxSwift: return "MLX Swift"
        case .llamaCpp: return "llama.cpp"
        case .coreMLLLM: return "CoreML (swift-transformers)"
        case .mediaPipe: return "LiteRT-LM"
        case .executorch: return "ExecuTorch"
        case .anemll: return "ANEMLL (ANE)"
        case .appleFM: return "Apple Foundation Models"
        case .coreAI: return "Apple Core AI"
        case .cactus: return "Cactus"
        }
    }
}
