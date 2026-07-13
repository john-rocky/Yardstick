# Apple Silicon 上の litert-community / official-converter 製 LLM — Core AI vs MLX vs LiteRT-LM

**LiteRT-LM チーム向けの、フレームワーク横断 decode/prefill ベンチマーク。** `falcon3-comparison.md`（Falcon3 のみを対象としていた）を、次の 2 つのモデル群に拡張したもの:

1. **litert-community（未ベンチ）** — Google 公式の `litert-community/*` `.litertlm` モデルのうち、まだ Core AI / MLX との比較がなかったもの: **DeepSeek-R1-Distill-Qwen-1.5B, Phi-4-mini,
   Gemma3-1B-IT, TinySwallow-1.5B, VibeThinker-1.5B**。
2. **我々による official-converter での変換** — upstream の `litert-torch`（BOCTAV4 int4）で変換した dense モデル: **OLMo-2-1B, Qwen3-1.7B, Llama-3.2-3B, SmolLM3-3B, Ministral-3-3B**。

**ハードウェア:** Mac (M4 Max GPU) + iPhone 17 Pro（オンデバイス）。**プロトコル (Mac):** 512 生成トークンでの定常状態 decode、greedy。Core AI は `llm-benchmark -p 512 -g 1024 -n 5`、MLX は `mlx_lm` の 512-tok ストリーム、LiteRT-LM は `litert-mac-verify --max-tokens 512 --backend gpu`。**プロトコル (iPhone):** short-chat、128 トークン、greedy、3 回のコールド起動の中央値（Yardstick アプリ）。日付: 2026-06-23。著者: john-rocky。

**条件（プラットフォーム / ランタイム別）:**

| Platform | Runtime | Prompt / prefill | Decode (greedy, temp 0) | Repeats |
|---|---|---|--:|---|
| iPhone 17 Pro | Core AI / MLX / LiteRT-LM | `"Explain what on-device AI means in simple terms."` (~20–30 tok incl. chat template) | **128 tok** | median of 3 cold |
| Mac M4 Max | Core AI | 512-token synthetic prompt | **512 tok** | n=5 |
| Mac M4 Max | MLX | short prompt | **512 tok** | — |
| Mac M4 Max | LiteRT-LM | short prompt | **512 tok** | — |

decode はランタイム間で **iso**（プラットフォームごとに同じ生成トークン数 + greedy、注記がない限り同じ int4 重みサイズ）→ decode tok/s の比較は同条件（apples-to-apples）である。**Mac の *prefill* プロンプトはまだ iso ではない**（Core AI は 512 トークンのプロンプト、MLX/LiteRT は短いプロンプトを使用）→ これは **prefill tok/s にのみ** 影響し（方向性の参考のみ、後述）、decode には影響しない。現在は 1 つの共通プロンプトへの標準化を進めている。iPhone の表は完全に iso である。

> **cold / warm（2026-07-13 更新、fairness-rules §2）。** 本レポートの iPhone 数値は **cold**
> （新規プロセス・初回生成）であり、ベンダーのモデルカード（in-process 定常 = warm）とは直接
> 比較できない。Qwen3 0.6B/1.7B/4B と Gemma-4-E2B の iPhone セルは 2026-07-13 に warm 併記で
> 再計測済み（`results/raw/2026-07-13-iphone-warm/`、warm = `--runs 4` の runs 2-4 中央値）。
> 詳細は英語版の "Warm re-capture (2026-07-13)" 節を参照。要点: **≥1.7B では順位変動なし**、
> 唯一の訂正は 0.6B の「LiteRT ≈ MLX タイ」で、これは MLX の Debug ビルドとの比較だった —
> Release warm 同一セッションでは **MLX 158.8 vs LiteRT 120.4（LiteRT ≈ 0.76×）**。他 9 モデルの
> Core AI セルはツールチェーン regression（`methodology/coreai-build-regression-2026-07.md`）で
> 再組み立て不能のため cold ラベルのまま。クロスセッション比較は本デバイスでは無効
> （`results/raw/2026-07-13-mlx-variance/`）。

> ⚠️ **量子化は均一ではない — これが中心的な注意点である。** MLX と Core AI は
> **4-bit**（モバイル標準の重みサイズ）でベンチマークしている。LiteRT-LM については **litert-community
> が実際に公開しているファイル** をベンチマークしており、これはモデルごとに異なる: **DeepSeek-R1 と Phi-4-mini は INT8 (q8) のみで提供** され、
> Gemma3-1B は INT4 で提供される。decode はメモリ帯域幅律速であるため、INT8 モデルはトークンあたり約 2 倍のバイトを読み出し、
> 本質的に INT4 の約半分の tok/s になる — **int8 の LiteRT 行は int4 の MLX/Core AI 行と
> 直接は比較できない。** 我々自身の変換はすべて INT4 (BOCTAV4) なので、それらの行は iso-bit *である*。

---

## ヘッドライン — Mac M4 Max GPU、decode tok/s（注記がない限り 4-bit、高いほど良い）

| Model | LiteRT quant | **Core AI** | **MLX** | **LiteRT-LM** |
|---|---|--:|--:|--:|
| **— litert-community —** | | | | |
| DeepSeek-R1-Distill-Qwen-1.5B | int8* | **319.5** | **323.6** | **115.9*** |
| Phi-4-mini-instruct | int8* | ✗ partial-RoPE¶ | **167.4** | **67.7*** |
| Gemma3-1B-IT | int4 | **327.2**† | **345.5** | **185.7**‡ |
| TinySwallow-1.5B | int8* | **324.1** | **326.7** | **119.6*** |
| VibeThinker-1.5B | int8* | **322.7** | **176.3**§ | **119.6*** |
| **— our official-converter (all int4 BOCTAV4) —** | | | | |
| OLMo-2-1B | int4 | **384.4** | **413.3** | **140.2** |
| Qwen3-1.7B | int4 | **239.1** | **322.7** | **115.8** |
| Llama-3.2-3B | int4 | **198.3** | **208.0** | **94.0** |
| SmolLM3-3B | int4 | **192.9** | **196.0** | **91.2** |
| Ministral-3-3B | int4 | **186.0** | **189.0** | **92.8** |

§ VibeThinker の MLX は、同じパラメータ数にもかかわらず他の 1.5B Qwen 系モデルより約 1.8× 遅く decode する（176 vs ~320）— おそらく config 内の intermediate/vocab がより大きいためで、再確認のためにフラグを立てている。

‡ Gemma3-1B LiteRT = **自前で変換**（google/gemma-3-1b-it → 公式 `litert_torch` で int4 BOCTAV4 に変換。litert-community のミラーは HF-gated）。1B + gemma 向けにチューニングされた LiteRT カーネルにより **185.7** で decode する — より大きな汎用 int4 ビルド（OLMo-2-1B 140, Qwen3-1.7B 116, Llama-3.2-3B 94）より *速い* 。これは `litert-speed-findings.md` の「Gemma は first-class な LiteRT カーネルの扱いを受けるが、汎用のサードパーティ経路は受けない」というテーゼを直接示す例である。ここでは MLX との差も相応に小さい（345 / 186 = 1.86×）。

`*` int8 = litert-community がそのモデルについて公開している唯一の quant。int4 の MLX/Core AI セルとは iso-bit ではない。
`†` Gemma3-1B Core AI は `coreai-models/export/pipeline.py` へのローカルな text-only-config 修正による（そうしないと gemma3 ラッパーは
multimodal config を前提としてしまう）。**2026-06-25 の export pass** で **olmo2 / smollm3 /
ministral3** 用の macOS クラスを追加した（ベンチ済み — 384.4 / 192.9 / 186.0）→ **Mac Core AI は現在 9/10**。`¶` **partial-RoPE** = Phi-4-mini の
`partial_rotary_factor` 0.75 は 64 head-dim のうち 48 を回転させる。export は HF parity を通るが、Mac の `llm-benchmark`
ハーネス（`RoPE freqs [48] ≠ half_embed [64]`）も iOS パイプライン（`Array index out of range`）も full rotary を前提としている —
Mac に残る唯一のブロックであり、iPhone **GPU + ANE** では hard wall となる。`△` Gemma3-1B ANE = local/global の dual RoPE +
sliding/full attention が、single-mask / single-RoPE の iOS パイプラインでは表現できない（GPU は問題なし）。`◇` Ministral-3-3B ANE
= multimodal-FP8 の `Mistral3ForConditionalGeneration` loader が必要（GPU は macOS-shim 経路で提供される）。

## Prefill tok/s (Mac M4 Max GPU)
| Model | Core AI | MLX† | LiteRT-LM |
|---|--:|--:|--:|
| DeepSeek-R1-1.5B | **4210** | 168† | 486 |
| Qwen3-1.7B | **3640** | 1004† | 396 |
| OLMo-2-1B | **4975** | 1491† | 672 |

† MLX/LiteRT の prefill は短いプロンプトを使用している（Core AI が使う 512-tok の合成プロンプトではない）ため、prefill はランタイム *内* での比較 / 方向性の参考のみとすること。いずれにせよ Core AI の prefill のリードは現実かつ大きい。

---

## 所見（暫定 — Mac）

1. **Apple GPU 上では Core AI ≈ MLX で、decode では両者とも LiteRT-LM を明確に上回る** — Falcon3 の調査と
   同じ結論であり、今回はより多くのアーキテクチャで再現された。iso-bit の int4 行について: Qwen3-1.7B = Core AI
   239 / MLX 323 / LiteRT 116、OLMo-2-1B = Core AI 384 / MLX 413 / LiteRT 140。LiteRT-LM は同じ int4 重みサイズで、decode において **MLX の約 35–50 %**
   に位置する。
2. **int8 のみの community モデルはさらに遅く見える — だがこれは公開時の選択であって、ランタイムの欠陥ではない。**
   DeepSeek-R1 と Phi-4-mini は litert-community 上では **q8/int8** でしか提供されていない — int8 ではバイト数が 2× になるため、
   decode は int4 ビルドの約半分になる。これはほぼ確実に意図的な品質判断であり（int4 PTQ は R1 のような
   推論モデルに最も強く影響する）、そのため我々は **int4 を指定するのではなく quant を開示する** ことにしている: MLX/Core AI との差の
   一部は公開された quant によるものであり、ランタイムによるものではない。
3. **Core AI の prefill は別格である**（3.6–4.2 k tok/s vs LiteRT の約 0.2–0.7 k）— Falcon3 と一致する。
4. **根本原因は `litert-speed-findings.md` から変わっていない:** LiteRT-LM は int4×int8 の **INTEGER** matmul を
   **WebGPU(Dawn)→Metal** デリゲート上で実行する。一方 MLX/Core AI は int4→fp16 に dequantize して **native-Metal の fp16 GEMM** を実行し、
   これは Apple GPU が極めて高速に処理する。decode の差は bit 幅ではなく、カーネル / デリゲートの効率による。
5. **Core AI のアーキテクチャ・カバレッジが実務上の制約だった — 2026-06-25 の export pass がその大半を解消し、
   「アーキごとのクラスが必要なだけ」というテーゼを裏付けた。** qwen2/qwen3/mistral ラッパーは DeepSeek-R1 (319.5)、TinySwallow
   (324.1)、VibeThinker (322.7)、Qwen3-1.7B (239.1) をきれいにカバーする。Gemma3-1B は既存の gemma3
   ラッパーへの text-only-config 修正だけで済んだ（現在 327.2 — それまでは multimodal の `text_config` / `language_model.` レイアウトを前提としており、text-only の
   1B で `AttributeError` を起こしていた）。続いてこの pass で **olmo2 / smollm3 / ministral3 用の不足していた macOS クラスを記述した** — いずれも
   きれいにベンチでき（**384.4 / 192.9 / 186.0**）、予測どおりだった（「未記述のクラスであって、未対応ではない」）。**Mac Core AI は現在
   9/10。** 唯一のブロックは **Phi-4-mini の partial rotary**（`partial_rotary_factor` 0.75）である: そのクラスは export でき
   HF parity も通るが、`llm-benchmark` の RoPE が full rotary を前提として中断する（`RoPE freqs [48] ≠ half_embed [64]`）— これは
   partial-rotary に対するハーネス / エクスポーターの制限であって、Core AI が「できない」アーキではない（同じ wall が iPhone GPU+ANE で再発する）。
6. **1.5B の Qwen 系モデルはいずれも Core AI と MLX の両方で約 320 tok/s で decode する**（DeepSeek 319.5/323.6,
   TinySwallow 324.1/326.7, VibeThinker 322.7/—）— 同じアーキテクチャ、同じ Apple-GPU の上限である。Core AI と MLX
   は統計的に同等で、LiteRT-LM の公開 int8 ビルドはその約半分に位置する（ランタイムではなく帯域幅による）。

## iPhone 17 Pro（オンデバイス）— Core AI vs MLX vs LiteRT-LM（short-chat、128 tok、3 コールドの中央値）

| Model | **Core AI ANE** | **Core AI GPU** | **MLX** | **LiteRT** |
|---|--:|--:|--:|--:|
| DeepSeek-R1-1.5B | **83.3** | **75.9** | **73.0** | **30.7** |
| TinySwallow-1.5B | **74.8** | **75.0** | **71.6** | **30.6** |
| VibeThinker-1.5B | **71.5** | **75.7** | — | **30.4** |
| Qwen3-1.7B | **64.8** | **67.6** | **62.8** | **23.2** |
| Qwen3-4B (iso-int4) | **29.7** | **28.3** | **27.3** | **21.2** |
| Gemma3-1B | ✗ dual-RoPE△ | **103.6** | **97.6** | gated |
| Phi-4-mini | ✗ partial-RoPE¶ | ✗ partial-RoPE¶ | **29.6** | OOM |
| OLMo-2-1B | **95.6** | **86.1** | — | **24.6** |
| Llama-3.2-3B | **24.2** | **19.3** | **34.0** | **18.4** |
| SmolLM3-3B | **23.0** | **20.5** | **36.8** | **22.8** |
| Ministral-3-3B | ✗ multimodal-fp8◇ | **17.6** | ✗ | **18.0** |

**オンデバイスでは結果はサイズとアーキに依存する — だが Core AI の ANE は、MLX/LiteRT が構造的に
使えない切り札であり続ける。** Core AI iOS がカバーする qwen-arch の ≤1.7B では Core AI がリードする: DeepSeek-R1 **ANE 83.3** / GPU 75.9 vs MLX 73.0 vs
LiteRT 30.7、TinySwallow 74.8 / 75.0 vs 71.6 vs 30.6、Qwen3-1.7B 64.8 / 67.6 vs 62.8 vs 23.2 — **Core AI は MLX と同等か上回り
（DeepSeek-R1 では ANE が MLX を上回る、83 vs 73）、両者とも LiteRT-LM の約 2.5×。** **2026-06-25 の export pass で iPhone にさらに 6 モデルが追加され**、
結果は **サイズで分かれる**: **1B** では Core AI が先行する — **Gemma3-1B GPU 103.6 > MLX 97.6**、また
**OLMo-2-1B ANE 95.6 / GPU 86.1 ≈ LiteRT 24.6 の 4×**（MLX リポジトリなし）。**3B では MLX が Core AI を上回る**: Llama-3.2-3B
MLX 34.0 vs Core AI ANE 24.2 / GPU 19.3（≈ LiteRT 18.4）、SmolLM3-3B MLX 36.8 vs Core AI ANE 23.0 / GPU 20.5（≈ LiteRT
22.8）、Ministral-3-3B Core AI GPU 17.6 ≈ LiteRT 18.0（MLX はロードできない）。**どのサイズでも 2 つの定数が成り立つ:** (1)
**Core AI 内では ANE が常に自身の GPU を上回る**（OLMo-2 95.6 > 86.1, Llama 24.2 > 19.3, SmolLM3 23.0 > 20.5）— そして
**ANE には Core AI / CoreML 経由でしか到達できない**（MLX と LiteRT-LM は Apple では GPU 専用）ため、最も
電力効率の良い経路でもあり、Mac では見えない Core AI の真のオンデバイスの強みである。(2) **Core AI の両経路はいずれも LiteRT-LM 以上を保つ。**
したがって正直なオンデバイスのまとめは **≤1.7B-qwen と 1B では Core AI ≥ MLX ≫ LiteRT、3B では MLX > Core AI ANE > GPU ≈ LiteRT**
である — MLX の成熟した Metal decode カーネルが帯域幅律速の 3B 領域を制し、一方 Core AI は ANE と小さいモデルを制する。

**iPhone の Core AI セル 3 つは、未着手の作業ではなく、確認済みのアーキテクチャ上の wall である**（正当な Core AI iOS パイプラインの
カバレッジに関する所見）: **Phi-4-mini の partial rotary**（`partial_rotary_factor` 0.75）は **GPU と ANE の両方** で `Array index out of
range` でクラッシュする — Mac の `llm-benchmark` の `RoPE freqs [48] ≠ half_embed [64]` 失敗と同じ根本原因であり、標準パイプラインは
full rotary を前提としている。**Gemma3-1B ANE** は、交互の sliding/full attention + local/global の dual RoPE を
single-mask/single-RoPE の iOS パイプラインでは表現できない（GPU は問題なし — 103.6）。**Ministral-3-3B ANE** は multimodal-FP8 の
`Mistral3ForConditionalGeneration` loader を必要とする（GPU は macOS-shim 経路で提供 — 17.6）。（前セッションの注記: VibeThinker の
ANE バンドルを再構成して **71.5 tok/s** を計測した。`ministral3`/OLMo-2/VibeThinker には mlx-community の iPhone リポジトリが存在しない。）

**iPhone の 3B int4 は LiteRT-LM で問題なくロードできる — 以前の失敗は LiteRT ではなく *ハーネス* の設定ミス
（メモリ entitlement の欠如）だった。** BenchmarkApp は `com.apple.developer.kernel.increased-memory-limit`
と `…extended-virtual-addressing` なしでビルドされていた。これらがないと、この **12 GB** デバイスでも大きな重みセクションの `mmap` が `Cannot allocate memory`
（ENOMEM）で失敗する — **Ministral-3-3B 0–1/3、Llama-3.2-3B 1/3** という結果だった。**両方の
entitlement を追加して解決した: Ministral-3-3B 3/3（~18 tok/s）、Llama-3.2-3B 5/5（~18.4）。** 両方が必要である —
`increased-memory-limit` はフットプリントの上限を引き上げ（小さなセクションが多数あるケース、Ministral）、
`extended-virtual-addressing` は 1 つの大きな連続セクションのためのアドレス空間を供給する（Llama）。途中で 2 つの誤った仮説を
追った — `externalize_embedder` のバグ、次に「LiteRT mmap ローダーの制限」 — **どちらも否定された。**
MLX が「LiteRT にできなかった 3B をロードできる」ように *見えた* のは、ロードピークが低く既定の上限内に収まっていたからにすぎない。
**iPhone の 3B 失敗は LiteRT-LM のせいではなく、我々のアプリ設定だった。**（教訓: iPhone のハーネスは ≳2 GB のモデルには
両方の entitlement が必要。続いて Phi-4-mini LiteRT を再確認した: その **int8 ビルド（3.6 GB）は entitlement が *あっても*
OOM する**（`std::bad_alloc` → SIGABRT）— だがこれは **quant の問題であって、Phi や LiteRT の問題ではない**。MLX は Phi を
int4（~2 GB, 29.6 tok/s）で実行し、LiteRT は他の 3B int4 ビルド（~2.2 GB）を問題なく実行するので、LiteRT の int4 Phi（~2 GB）も
ロードできるはずだ — litert-community が Phi を int8 でしか提供していないだけである。したがって iPhone の「OOM」セルは次のように分かれる: **3B int4 = ハーネスの
副産物（現在はロードできる）。Phi int8 = *その quant* では大きすぎる（int4 なら収まる）。オンデバイスで本当に実行不可能なもの = なし。**）

**Core AI iOS:** 現在 **16/20** セルを計測済み — qwen-arch の ≤1.7B セット（DeepSeek-R1, TinySwallow, VibeThinker,
Qwen3-1.7B）**に加えて 2026-06-25 の export pass**（Gemma3-1B GPU。OLMo-2-1B, Llama-3.2-3B, SmolLM3-3B はいずれも ANE+GPU。
Ministral-3-3B GPU）。≤1.7B-qwen と 1B では Core AI ≥ MLX ≫ LiteRT、**3B では MLX が先行**、**全域で ANE > Core AI 自身の GPU**。

### カバレッジ — 計測セル 60 個。空欄はすべて文書化されたアーキテクチャ上のブロック
- **Mac 29/30:** 10 モデルすべてが MLX + LiteRT を持つ。**Core AI 9/10**（qwen2/qwen3/mistral/gemma3 + 新規の
  olmo2/smollm3/ministral3 クラス — 384.4/192.9/186.0）。Core AI で唯一のブロックは **Phi-4-mini の partial-RoPE**（export でき
  parity も通るが、`llm-benchmark` の full-rotary RoPE が中断する）。
- **iPhone 31/40:** Core AI **16/20**（7 ANE + 9 GPU）、MLX 7/10、LiteRT 8/10。ブロック: **Core AI ANE ×3** — Gemma3-1B
  （dual-RoPE/sliding）、Phi-4-mini（partial-RoPE）、Ministral-3-3B（multimodal-fp8 loader）— **加えて Phi GPU**（partial-RoPE、
  唯一の GPU ブロック）。**MLX ×3** — OLMo-2/VibeThinker には mlx-community リポジトリがなく、MLX-Swift は `ministral3` をロードできない。
  **LiteRT ×2** — Gemma3-1B は gated、Phi int8（3.6 GB）は OOM。（Llama/Ministral の 3B LiteRT は、メモリ entitlement を付けることで現在はロードできる。）

## Supplementary — LiteRT int4（byte-vs-delegate の切り分け）+ Core AI static-GPU の状況

**LiteRT の int8 vs int4 — 推定ではなく実測。** 公式の litert-community DeepSeek-R1-1.5B は **q8**（int8）で提供される。
同じモデルの **バイト一致の int4**（自前の BOCTAV4 blockwise-32 + OCTAV、GSM8K 73%）を補足行として
追加した。公式の q8 行はそのまま残す。iPhone 17 Pro、short-chat、3 コールドの中央値:

| DeepSeek-R1-1.5B · LiteRT-LM · iPhone | decode tok/s | peak MB | GSM8K (bf16 = 81.0) |
|---|--:|--:|--:|
| q8 (official litert-community, ≈quality-parity) | 30.7 | ~1,700 | ≈parity |
| int4 (own BOCTAV4, **reference only — not parity**) | **45.0** | 1,250 | **73.0 (−8 pt)** |

⚠ **int4 の行は速度の _参考値_ であって、品質パリティの成果物ではない。** GSM8K は 73.0% で、bf16 の 81.0% に対し −8 pt
（1.5B 推論モデルの 4-bit 感度。7B の同系列は −1 pt でパリティ）。したがってバイトの効果は
「int4 は 45.0 に *到達しうる* が、精度を 8 pt 犠牲にする」と読むべきで、無償の高速化ではない。公式の **q8 がパリティ行である。**

その注意点を踏まえても、この切り分けは依然として示唆に富む。int8→int4 のバイト削減は **1.47×**（30.7 → 45.0）を生む — 現実の効果だが、
純粋に帯域幅律速のモデルが与えるはずの ~2× には届かず、**品質の境界をまたぐ**。よりクリーンな **iso-int4** の
比較は 1 つの quant 内に留まる: **LiteRT int4（45.0）は Core AI / MLX の int4（~73–83）のわずか ~0.6× にすぎない** — その残差の
**~1.67×** が **delegate/カーネルの差**（WebGPU(Dawn)→Metal vs native-Metal fp16 GEMM）であり、4-bit を揃えて計測したものである。つまり
LiteRT-LM のオンデバイスの差は **int8 のバイト**（~1.5×、精度コストあり）**と** **delegate**（~1.7×、iso-int4、品質中立）の **両方** である —
「単に int8 だから」では説明がつかず、native-Metal カーネル経路の方が大きく、よりクリーンなてこである。

**Core AI static-GPU の 3-way — 先送り（coverage gap）。** ANE-vs-GPU の *エンジン* を、static-shape と palettized quant を揃えて
切り分けるための static-shape-on-GPU export を計画し、10 個の dense モデルすべてについてコンパイルした（`*_static_gpu`、ANE 領域が 0 であることを検証済み）が、
アプリの Core AI ランタイムでは **ロードできない**: すべてのエンジン variant（`static-shape` /
`coreai-pipelined` / auto）が `EngineFactory` で POSIX Code 2 `No such file` で失敗する — GPU コンパイルは
`mpsExecutable.mpsgraphpackage`（original/specialized モデル + resources.bin）を出力するが、static エンジンが消費する per-bucket の
`binary_0.llir.bundle/…/extend_*` アーティファクトを欠いている。修正は export 側（GPU バケットの完全な
specialization、または gemma4-bucketed への移植）であり、**別セッションに先送りした**。したがって ANE-vs-GPU の decode 行は
記載どおりに成立する: ≈ タイで、モデルによって符号が反転する（DeepSeek は ANE>GPU、Qwen3-1.7B/VibeThinker は GPU>ANE）= これは
export-shape + cold-launch の効果であって、**安定したエンジンの順位ではない** — そして ANE の擁護可能な強みは依然としてエネルギー +
排他性（MLX/LiteRT は ANE をまったくターゲットにできない）であり、生の decode tok/s ではない。

## エネルギー — iPhone 17 Pro、持続的な decode（battery-delta、J/token）

上の decode の表は short-chat のスナップショットである。この軸はオンデバイスの **効率** に関する問い、すなわち
*持続的な負荷の下でジュールあたり最も多くのトークンを生み出すランタイムはどれか?* を扱う。計測手段: `UIDevice.batteryLevel`
（1% 刻み）を、ランタイムを稼働させ続けるために再プロンプトする **600 s の `energy` タスク** にわたってサンプリングする。約 5% の低下 →
デバイスのバッテリーパック（iPhone 17 Pro = 16.5 Wh）からジュールに換算する。システム全体（ディスプレイ + 無線 + OS を含む）で、1% 解像度では **実行ごとに ±~12%**
— 比較は *同一デバイス内* に限ること。方法: [`energy-ios.md`](../methodology/energy-ios.md)。**3 モデル** — **Llama-3.2-3B**、**DeepSeek-R1-1.5B**、**Qwen3-4B**（後者 2 つは Google 自身の litert-community 配布物）— 4 つのランタイムすべて（注記がない限り 4-bit、unplugged、画面オン、バッテリー中域）:

**Llama-3.2-3B**（すべて int4）— ピークメモリ列（`memoryPeakDuringDecodeMB`）を追加:

| Runtime | **J/token** ↓ | avg W | sustained tok/s | (short-chat) | **peak MB** | tokens/Wh |
|---|--:|--:|--:|--:|--:|--:|
| **Core AI GPU** | **0.224** | 4.67 | 21.0 | 19.3 | **732** | 16 058 |
| **Core AI ANE** | **0.225** | 4.62 | 20.6 | 24.2 | 2,577 | 16 019 |
| **MLX** | **0.245** | 4.70 | 19.3 | 34.0 | 3,221 | 14 681 |
| **LiteRT-LM** | **0.324** | 4.31 | 13.6 | 18.4 | 3,263 | 11 120 |

**DeepSeek-R1-1.5B**（Core AI / MLX は int4。LiteRT int8* = その litert-community quant）:

| Runtime | **J/token** ↓ | sustained tok/s | (short-chat) | **peak MB** | tokens/Wh |
|---|--:|--:|--:|--:|--:|
| **Core AI ANE** | **0.097** | 49.0 | 83.3 | 1,368 | 37,236 |
| **MLX** | **0.105** | 44.8 | 73.0 | 1,434 | 34,315 |
| **Core AI GPU** | **0.132** | 36.2 | 75.9 | **374** | 27,307 |
| **LiteRT-LM** | **0.204** | 24.2 | 30.7 | 1,000 | 17,678 |

**Qwen3-4B**（すべて **iso-int4** — LiteRT = litert-community の公式 mixed-int4 なので、quant の交絡なし）:

| Runtime | **J/token** ↓ | sustained tok/s | (short-chat) | **peak MB** | tok/1% |
|---|--:|--:|--:|--:|--:|
| **Core AI ANE** | **0.242** | 17.4 | 29.7 | 3,181 | 2,458 |
| **MLX** | **0.242** | 18.2 | 27.3 | 3,996 | 2,458 |
| **Core AI GPU** | 0.290 | 16.5 | 28.3 | **873** | 2,048 |
| **LiteRT-LM** | **0.725** | 18.0\* | 21.2 | 2,217 | 819 |

\* LiteRT の decode *レート* は競争力がある（~18）が、その **実効** スループットは ~4.7 tok/s である — ウィンドウ内で生成したのは
**4,096** トークンにすぎず、ANE/MLX の 12,288 に対して少ない（re-prefill / オーバーヘッド支配、gen-time 865 s）→ **tok/1% は 3× 最悪** で、
3 モデル中で最大の差であり、iso-int4 では quant の言い訳が効かない。

**メモリ（decode 中のピーク）— Core AI GPU は別格:** 732 MB（Llama）/ 374 MB（DeepSeek）/ 873 MB
（Qwen3-4B）を mmap-pipelined な重みで実現する（resident set ~280 MB）。一方 MLX と LiteRT は ~3.2–4.0 GB、
重みが常駐する Core AI ANE は 2.6 / 1.4 / 3.2 GB である — pipelined-mmap 経路だけが生む **~4× のメモリ余裕の優位** であり
（より大きなモデルをオンデバイスに収めるうえで決定的）。

**3 モデルすべてで、Core AI が最もエネルギー効率が良く、LiteRT-LM が最も悪い**（Llama: Core AI ANE≈GPU
~0.224 vs MLX 0.245 vs LiteRT 0.324、DeepSeek: Core AI ANE 0.097 vs MLX 0.105 vs GPU 0.132 vs LiteRT 0.204、
Qwen3-4B: Core AI ANE 0.242 ≈ MLX 0.242 vs GPU 0.290 vs LiteRT 0.725）。すべての
ランタイムが同じ 5%（1 quantum）を消費したため、**J/token は *持続的な* スループットによって決まる**（ジュールは一定で、
トークン数が異なる）:
1. **持続的な負荷は short-chat のランキングを並べ替える — MLX が最も打撃を受ける。** MLX の 34 tok/s という short-chat のリードは、
   10 分間の熱負荷（ピーク `serious`）の下で **19.3** まで崩れ、Core AI を *下回る* 。LiteRT は 18.4→13.6。Core AI は最も
   熱的に安定している（ANE 24→20.6, GPU 19.3→21.0）。energy タスクは現実的な長文生成の領域である。
2. **エネルギーでは ANE ≥ GPU。GPU の強みはメモリ（エネルギーではない）。** Core AI ANE は、どのサイズでも J/token で GPU に
   勝つか同点である — DeepSeek-1.5B（ANE 0.097 vs GPU 0.132）、Qwen3-4B（0.242 vs 0.290）、そして Llama-3B では同点（0.225 ≈
   0.224、GPU がたまたま異例に安定して持続したケース）。これはきれいにサイズ依存というわけ **ではない**。GPU が一貫して勝つのは
   **メモリ** である: その mmap-pipelined 経路は **374–873 MB** を保つのに対し、ANE の重み常駐は **1.4–3.2
   GB** である — ~4× の余裕の優位。したがって **ANE = 効率の選択、GPU = メモリの選択** である。システム全体の電力は
   ディスプレイ/OS が支配的（~4.6 W）なので、バッテリー数値ではどちらもチップ電力の優位を持たない。
3. **「低電力だが遅い → 最悪の J/token」という役回りは、ここでは LiteRT が担う**（最小の消費 4.31 W だが最も遅い 13.6
   tok/s → 最悪の 0.324）— *Mac 上で* ANE が示したのと同じ形である（後述）。

**このリポジトリの過去の energy 行との比較**（`results/raw/*-energy-*.jsonl`）:
- **Mac でのパターンはスマートフォンでは反転する。** **M4 Max** では、CoreML/ANE の gemma-4-E2B はワット数が *最小*（12.7 W）でありながら
  J/token は *最悪*（**0.478**）だった — その 33 tok/s がパッケージを最も長く通電させ続けたためである。MLX（0.240）と llama.cpp（0.247）が
  速度で勝った。**iPhone ではこれが逆転する:** Core AI が *最良* である。GPU 系ランタイムはオンデバイスで急峻な
  throttling 税を払い（MLX 35→19）、システム全体の消費が電力を均等化するためだ。これはまさに
  `energy-ios.md` が提起した問いであり、答えは **「はい、オンデバイスのスループット税はエネルギーの勝者を変える」** である。
- **効率はランタイムで絶対的に決まるのではなく、アーキテクチャ固有である。** **iPhone の gemma-4-E2B では LiteRT-LM が
  *最良*（0.146 J/token）** だった — その first-class な gemma カーネルが持続 30.8 tok/s で decode するためだ — 一方 Llama-3.2-3B
  （汎用経路）では LiteRT-LM が *最悪*（0.324）である。つまり「どのランタイムが最もグリーンか」はモデルが
  first-class なカーネルを持つかどうかに依存し、decode 速度のテーゼを反映している。
- **絶対値はモデルサイズが支配する。** MLX 上の qwen3-0.6B は **0.066 J/token**（99 tok/s、`fair` を維持）に達した —
  どの 3B セルよりも約 4× 効率が良い。より小さいモデルこそ最大のエネルギーのてこであり、ランタイムの選択よりも効く。

**「iPhone のバッテリー 1% で何トークン買えるか?」** ヘッドライン指標（トークン ÷ バッテリー Δ%）として捉え直すと、
**バッテリー効率の王座はモデルによって入れ替わる** — ランタイムではなく、first-class カーネルのサポートを追従する:

| Model (LiteRT quant) | **LiteRT-LM** | **Core AI** (ANE / GPU) | **MLX** | greenest |
|---|--:|--:|--:|---|
| Gemma-4-E2B (int4, gemma kernels) | **4,074** | 2,867 (ANE) | 2,560 | **LiteRT-LM** |
| DeepSeek-R1-1.5B (int8*, generic) | 2,917 | **6,144** / 4,506 | 5,662 | **Core AI (ANE)** |
| Llama-3.2-3B (int4, generic) | 1,835 | 2,643 / **2,650** | 2,422 | **Core AI (GPU)** |
| Qwen3-4B (int4, generic) | 819 | **2,458** / 2,048 | 2,458 | **Core AI / MLX** |

バッテリー 1% あたりのトークン数 — **バッテリー効率の *ランキング* はモデルによって入れ替わる。** **Gemma — Google の first-class な
モデル — では、LiteRT-LM が Apple のスマートフォン上で最もグリーンなランタイムである（4,074 tok/1%）**。**汎用モデルでは最下位に落ち、
Core AI が勝つ** — これは **3 つの独立した非 gemma モデル** で確認された: Google 自身の **DeepSeek-R1-1.5B**（Core AI
ANE **6,144** = LiteRT の 2,917 の **2.1×**）、**Llama-3.2-3B**（Core AI 2,650 vs LiteRT 1,835）、そして **Qwen3-4B** —
4 つのランタイムすべて **iso-int4** — で LiteRT は **3× 最悪**（Core AI/MLX 2,458 vs LiteRT **819**）。この入れ替わりは
明白な交絡に対して頑健である — LiteRT が **int4**（Llama, Qwen3-4B）で提供されようと **int8**（DeepSeek）で提供されようと、また行が
Debug/iOS 26.4.2（Gemma）であろうと Release/iOS 27.0（DeepSeek, Llama）であろうと成り立つ — したがって原因は **first-class カーネルの差** であり、
quant やビルドの副産物ではない（LiteRT は Gemma を持続 30.8 tok/s で decode するのに対し、DeepSeek は 24.2、Llama は 13.6）。**Google
自身が litert-community 経由で配布しているモデル（DeepSeek-R1）でさえ、LiteRT-LM は 4 つの中で最もバッテリー効率が悪い。**（絶対の
tok/1% は、より小さい DeepSeek-1.5B では上がる — モデルサイズが最大のてこである — ので、行をまたいだ大きさではなく、行 *内* の
*ランキング* を比較すること。）

> ⚠️ 単一実行、1% バッテリー解像度: ±0.02–0.03 J/token の差（ANE vs GPU）はタイとみなすこと。MLX と LiteRT の
> 差は誤差棒より大きい。すべての行で `batteryState=unplugged` と非 nil の低下を確認済み。

## カバレッジ — LiteRT は *トレースする*、Core AI は *再実装する*（すぐ使える手軽さ vs カスタムコードの天井）

ここで扱うモデルはすべて **カスタムコードなし** で LiteRT に変換できた。一方 Core AI はアーキテクチャごとに手書きのクラスが必要なので、
2026-06-25 の export pass の前は 6/10 しかカバーしていなかったが、3 つのクラスを追加で書いたことで **現在は Mac で 9/10** になった。これは 2 つのコンバーターの設計の
直接的な帰結であって、品質の差ではない。**LiteRT-LM（`litert_torch`）はトレースベースでアーキテクチャ非依存である:** `torch.export` が HF モデルの *実際の* forward を
捕捉し、それを汎用 op（matmul, RMSNorm, 実数値 RoPE, softmax-SDPA）に下ろす。アーキテクチャを「知る」必要はまったくない — OLMo-2 の
QK-norm、SmolLM3 の NoPE、Ministral のレイアウトは *同じ汎用 op の異なる配置にすぎない* ため、サポートされた op を持つトレース可能なモデルなら何でも動作する。**Core AI（`coreai-models`）はアーキテクチャごとの再実装
レジストリである:** Apple のプリミティブ（SDPA/RoPE/RMSNorm）から構築した手書きの Apple `nn.Module` クラス（`macos/{qwen2,qwen3,mistral,gemma3_text,…}.py`）に HF の重みが *ロード* される（`_mutate_state_dict` は
QKV の再結合まで行う）。手書きクラスのないアーキテクチャは export できない — これが olmo2/smollm3/ministral3 に元々
クラスがなかった理由である（そして `mistral` ラッパーを流用することもできない: *動きはする* が誤った計算をしてしまう — QK-norm をスキップしたり、NoPE 層に
RoPE を適用したりする）。**export pass はそれらのクラスを書き、いずれも動作する**（Mac 384.4/192.9/186.0、iPhone でも）。差は「未記述のクラス」であって「未対応」ではなかったことが確認された。残る 2 つの wall は、クラスの欠如よりも深い:
**Phi-4-mini の partial rotary**（クラスは存在し HF parity も通るが、標準の full-rotary RoPE 経路が Mac の `llm-benchmark` と iOS パイプラインの両方で中断する）、
および **Gemma3-1B / Ministral-3-3B の ANE** セル（iOS ANE パイプラインは Gemma3 の dual-RoPE + sliding/full attention を表現できず、Ministral は multimodal-FP8 loader を必要とする —
どちらも GPU 経路では問題なく動作する）。Gemma3-1B が早い段階の手早い修正だったのは、gemma3 が *すでに* レジストリにあったからである —
config レイアウトの前提を緩めるだけで済んだ。

Core AI が（トレースではなく）再実装するのは、**ステートフル（KV-cache-as-state）で Apple 向けにチューニングされた memory-mapped
グラフ** を出力するためであり、これが速さ（320 tok/s, 4 k prefill）の源である。**重要なニュアンス（そしてありがちな
誇張の訂正）: すぐ使える状態では、dense モデルにとって LiteRT-LM の方が *手軽* である — カスタムコード不要 — だが、より広いカバレッジを
持つわけではない。** カスタムコードを許せば **Core AI の天井ははるかに高い**: 手書きの export クラス *と* カスタム Metal カーネルがあれば、
MoE、diffusion、SAM / depth-anything / YOLO / efficient-SAM、音声（whisper / wav2vec2 / clap）、T5 をオンデバイスで **到達でき、しかも高速に実行できる**
— `coreai-models` の zoo には 20+ のファミリーがある。LiteRT-LM は今日それに匹敵できない: 最新のアーキ（MoE / MLA / SSM）は **変換後でも CPU に
落ちて** しまい、その **GPU カーネル層は閉じている**（ML Drift はプリビルドで、外部のカーネル登録経路がない）。したがって LiteRT-LM の真の強みは
**摩擦ゼロの dense 変換 + 可搬性 / クロスハードウェアの到達範囲** であって、生のカバレッジの広さ *ではない* — そして高レバレッジな
要望は **天井を上げること: (a) 新アーキ向けのコンバーター修正、(b) Core AI を拡張するのと同じようにコントリビューターが拡張できるよう
GPU カーネル層を開放すること** であり、加えて dense 経路の Apple-GPU 速度差を埋めること（後述）である。（ここで扱う 10 個 — すべて dense —
のうち、LiteRT は **モデルごとのコードなし** で 10/10 を変換する。Core AI はアーキごとに手書きクラスを必要とするが、いったん書けば Mac で同じく **9/10**
を実行する。この *dense における手軽さ* の論点は、**新しいアーキテクチャの限界費用** — LiteRT はゼロ、Core AI はクラス 1 つ — であって、恒久的な
カバレッジの天井ではない。2026-06-25 の pass が示したとおりである。）

## LiteRT-LM チームへの提言（インパクト順）
1. **weight-only int4 + FLOAT 演算（int4→fp16 dequant + fp16 GEMM）を実際に動作させる** — 現状では
   変換はできるが WebGPU デリゲートでハングする（`deliverables/litert-float-gpu-hang/`）。これは MLX/Core AI の
   演算モデルであり、Apple-GPU の decode 同等性への単一で最もインパクトの大きい経路である。
2. Apple 向けの **ネイティブ Metal デリゲート**（WebGPU/Dawn→Metal に対して）で抽象化層を取り除く。
3. **Core AI を拡張するのと同じようにコントリビューターが LiteRT-LM を拡張できるよう、天井を上げる** — (a) 最新アーキ向けにコンバーターを
   修正する（MoE / MLA / SSM — 現状では変換後でも CPU に落ちる）、(b) **GPU
   カーネル層を開放する**（ML Drift はプリビルドで外部のカーネル登録経路がない）。これが、dense コンバーターが手軽であるにもかかわらず、現状で
   LiteRT-LM のオンデバイスのモデルセットを Core AI 以下に抑えている要因である。
4. **LiteRT の真の強みに振り切る** — クロスハードウェアの到達範囲（Android の Qualcomm/NPU）と 1 つの可搬なバンドル。
   Apple GPU では MLX/Core AI がネイティブの下限であり、（上回ることではなく）同等が現実的な目標である。

## 方法論 / 再現
**オープンなハーネス — 自分で再現できる: https://github.com/john-rocky/apple-silicon-llm-bench**（中立的で、すべてのランタイムに対する単一の
ヘッドレスハーネス。すべてのセルは raw JSONL まで辿れる）。ランタイム別:
- **Core AI:** `uv run coreai.llm.export <hf-id> --platform macOS --compression 4bit --compute-precision
  float16 --experimental`。ベンチは `llm-benchmark --model <bundle> -p 512 -g 1024 -n 5`。ラッパーは HF の
  `model_type` で選択される（`MODEL_TYPE_REMAPPING` 経由で qwen2/qwen3/gemma3/mistral）。
- **MLX:** `mlx-community/*-4bit`（または `mlx_lm.convert -q --q-bits 4 --q-group-size 64`）。512-tok の greedy ストリーム。
- **LiteRT-LM:** litert-community の `.litertlm`（公開 quant）または我々の BOCTAV4 変換。
  `litert-mac-verify --max-tokens 512 --backend gpu`。
- **iPhone 17 Pro（全ランタイム）:** Yardstick アプリ（`ios/BenchmarkApp`、バンドル
  `com.iosllmbenchmark.benchmarkapp`）を `devicectl … --yardstick-autorun --runtime
  {mlx-swift,litert-lm} --model-id <id> --task short-chat` でヘッドレスに駆動し、3 回のコールド起動の中央値をとる。モデルエントリは
  `ios/BenchmarkApp/Sources/Models/ModelCatalog.swift` に追加する（`mlx-community/*` = オンデバイスの MLX ダウンロード、
  `litert-community/*` = オンデバイスの LiteRT ダウンロードまたはサイドロード、`litert-local/*` = サイドロードした `.litertlm`）。
- **Raw decode セル（このリポジトリ）:** `results/raw/2026-06-24-litert-community-crossframework/decode-cells.jsonl`
  — model×framework×device ごとに 1 つの JSON 行: `{model, group, framework, device, quant, decode_tps, prefill_tps, …}`。
