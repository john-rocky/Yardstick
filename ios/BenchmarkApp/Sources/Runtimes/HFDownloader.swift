import Foundation
import HuggingFace

/// Shared helper that downloads a HuggingFace repo (or a subset of it) into
/// the app's `Documents/models/<runtime>/<repo-id>/` directory. Used by every
/// runtime adapter that does not have its own download mechanism.
public enum HFDownloader {
    public static func snapshot(
        for model: ModelInfo,
        runtime: RuntimeKind,
        progress: @Sendable @escaping (Double) -> Void
    ) async throws -> URL {
        let target = modelDirectory(runtime: runtime, hfRepoId: model.hfRepoId)

        // Already downloaded?
        if FileManager.default.fileExists(atPath: target.path),
           let contents = try? FileManager.default.contentsOfDirectory(at: target, includingPropertiesForKeys: nil),
           !contents.isEmpty {
            progress(1)
            return target
        }

        try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)

        guard let repoID = HuggingFace.Repo.ID(rawValue: model.hfRepoId) else {
            throw LLMRuntimeError.downloadFailed("Invalid HF repo id: \(model.hfRepoId)")
        }

        do {
            let downloaded = try await HubClient.default.downloadSnapshot(
                of: repoID,
                revision: "main",
                matching: model.hfFilePatterns,
                progressHandler: { @MainActor p in
                    progress(p.fractionCompleted)
                }
            )
            // The HubClient downloads into its own cache. Mirror primary file(s) into target.
            // For runtimes that need a stable path, use the HubClient cache directly.
            return downloaded
        } catch {
            throw LLMRuntimeError.downloadFailed(error.localizedDescription)
        }
    }

    /// The HF revision (commit hash) `hfRepoId` currently resolves to on this device,
    /// read from the hub cache's `refs/main` — the same file `stage` pins on a sideload.
    /// Best-effort: `nil` when the model did not come through a hub cache (sideloaded
    /// .litertlm / GGUF / CQ / .aimodelc bundles, whose identity is the file itself).
    public static func resolvedRevision(hfRepoId: String) -> String? {
        let cacheName = "models--" + hfRepoId.replacingOccurrences(of: "/", with: "--")
        var bases = [
            FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first?
                .appendingPathComponent("huggingface/hub", isDirectory: true),
            FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?
                .appendingPathComponent("huggingface/hub", isDirectory: true),
        ]
        #if os(macOS)
        // The Mac CLI's hub cache (python-hf layout). `homeDirectoryForCurrentUser` does not
        // exist on iOS — unguarded, this line silently failed the whole iOS build on
        // 2026-07-30 and an r3 binary got installed while the sources said r4.
        bases.append(FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub", isDirectory: true))
        #endif
        for base in bases {
            guard let ref = base?.appendingPathComponent("\(cacheName)/refs/main") else { continue }
            if let rev = try? String(contentsOf: ref, encoding: .utf8) {
                let trimmed = rev.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty { return trimmed }
            }
        }
        return nil
    }

    public static func modelDirectory(runtime: RuntimeKind, hfRepoId: String) -> URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return docs
            .appendingPathComponent("models", isDirectory: true)
            .appendingPathComponent(runtime.rawValue, isDirectory: true)
            .appendingPathComponent(hfRepoId.replacingOccurrences(of: "/", with: "__"), isDirectory: true)
    }
}
