// Mac parity check for converted .litertlm models — generate text on the Mac
// (no device push/launch), so we can iterate on the "degenerate output" bug fast.
//
//   swift run litert-mac-verify <model.litertlm> ["prompt"] [--max-tokens N]
//
// --max-tokens (default 256) lets reasoning models (<think>) finish their chain
// of thought and emit the actual answer instead of being truncated mid-thought.

import Foundation
import LiteRTFoundation
import LiteRTLM

// Parse: <model> [prompt] with an optional `--max-tokens N` / `-n N` flag anywhere.
var positional: [String] = []
var maxTokens = 256
var engineBackend: String? = nil  // --backend cpu|gpu → run via low-level Engine API
var engineTemp: Float = 0.0  // --temp T → sampler temperature for the Engine path (models whose greedy decode collapses, e.g. RL tunes, need their official sampling)
var engineTopK: Int = 40  // --topk K → sampler top-k for the Engine path
var greedy = false  // --greedy → temperature 0 (match HF do_sample=False) instead of LiteRTChat's default temp 0.8
var thinking = false  // --thinking → ThinkingConfig(enableThinking: true, budget -1); v0.15.0+ runtime only
var runs = 1  // --runs N → repeat generation N times in-process (run 1 = cold, 2+ = warm engine)
var i = 1
let argv = CommandLine.arguments
while i < argv.count {
  let a = argv[i]
  if a == "--max-tokens" || a == "-n", i + 1 < argv.count {
    maxTokens = Int(argv[i + 1]) ?? maxTokens
    i += 2
  } else if a == "--runs", i + 1 < argv.count {
    runs = Int(argv[i + 1]) ?? runs
    i += 2
  } else if a == "--backend", i + 1 < argv.count {
    engineBackend = argv[i + 1]
    i += 2
  } else if a == "--temp", i + 1 < argv.count {
    engineTemp = Float(argv[i + 1]) ?? engineTemp
    i += 2
  } else if a == "--topk", i + 1 < argv.count {
    engineTopK = Int(argv[i + 1]) ?? engineTopK
    i += 2
  } else if a == "--greedy" {
    greedy = true
    i += 1
  } else if a == "--thinking" {
    thinking = true
    i += 1
  } else {
    positional.append(a)
    i += 1
  }
}
guard positional.count >= 1 else {
  FileHandle.standardError.write(
    Data("usage: litert-mac-verify <model.litertlm> [prompt] [--max-tokens N]\n".utf8))
  exit(2)
}
let modelPath = positional[0]
let prompt = positional.count >= 2 ? positional[1] : "Explain on-device AI in one short sentence."

// Low-level Engine path with an explicit backend (--backend cpu|gpu) — used to
// localize where a recipe runs vs hangs (e.g. weight-only FLOAT compute).
if let bk = engineBackend {
  let backend: Backend = (bk == "cpu") ? .cpu() : .gpu
  let semE = DispatchSemaphore(value: 0)
  Task {
    do {
      ExperimentalFlags.optIntoExperimentalAPIs()
      ExperimentalFlags.enableBenchmark = true
      let t0 = Date()
      let config = try EngineConfig(modelPath: modelPath, backend: backend, maxNumTokens: maxTokens)
      let engine = Engine(engineConfig: config)
      try await engine.initialize()
      print(String(format: "engine[\(bk)] init %.1fs", Date().timeIntervalSince(t0)))
      let sampler = try SamplerConfig(topK: engineTopK, topP: 0.95, temperature: engineTemp)
      for r in 1...runs {
        let conv = try await engine.createConversation(with: ConversationConfig(samplerConfig: sampler))
        var yields = 0
        var acc = ""
        for try await chunk in conv.sendMessageStream(Message(prompt)) {
          yields += 1
          acc += chunk.toString
        }
        let oneLine = acc.replacingOccurrences(of: "\n", with: "⏎")
        if r == 1 { print("OUTPUT: [\(oneLine)]") }
        let b = try conv.getBenchmarkInfo()
        print("OK[\(bk)] run=\(r) cold=\(r == 1 ? 1 : 0) decode_tokens=\(b.lastDecodeTokenCount) stream_yields=\(yields)")
        print(String(format: "run %d: decode %.1f tok/s · prefill %.1f tok/s", r,
          b.lastDecodeTokensPerSecond, b.lastPrefillTokensPerSecond))
      }
    } catch {
      print("FAILED[\(bk)]: \(error)")
    }
    semE.signal()
  }
  semE.wait()
  exit(0)
}

let sem = DispatchSemaphore(value: 0)
Task {
  do {
    let t0 = Date()
    let chat = try await LiteRTChat(
      modelFileURL: URL(fileURLWithPath: modelPath),
      modalities: [] as Modality,
      maxTokens: maxTokens,
      enableBenchmark: true,
      sampler: greedy ? (try SamplerConfig(topK: 1, topP: 1.0, temperature: 0.0)) : nil,
      prewarm: false,
      thinking: thinking ? ThinkingConfig(enableThinking: true, thinkingTokenBudget: -1) : nil)
    print(String(format: "loaded in %.1fs", Date().timeIntervalSince(t0)))
    let resp = try await chat.respond(prompt)
    // brackets + single-line so an empty/whitespace response is visible and greppable
    let oneLine = resp.replacingOccurrences(of: "\n", with: "⏎")
    print("OUTPUT: [\(oneLine)]")
    if let b = try? chat.lastBenchmark() {
      print(String(format: "decode %.1f tok/s · prefill %.1f tok/s",
        b.lastDecodeTokensPerSecond, b.lastPrefillTokensPerSecond))
    }
  } catch {
    print("FAILED: \(error)")
  }
  sem.signal()
}
sem.wait()
