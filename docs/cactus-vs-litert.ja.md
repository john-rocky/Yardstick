# Cactus vs LiteRT-LM vs MLX — Apple silicon 上の Gemma-4-E2B

Cactus (YC S25) は Gemma-4-E2B-CQ4 の Apple デバイス性能表と、その再現手順
(`cactus benchmark --ios`) を[公開している](https://github.com/cactus-compute/cactus)。
本稿は、彼らが表に載せているのと同じ端末 — iPhone 17 Pro — での独立再現と、同じ端末・同じ
モデル・同じプロンプト長・同じデコード予算での LiteRT-LM および MLX との比較である。

以下はすべて本リポジトリのスクリプトが、Cactus の `main` と彼らの公開バンドル
`Cactus-Compute/gemma-4-E2B-it` @ `v2.0.1` に対して生成したものである。ラン単位の生 JSON は
[`results/raw/2026-07-10-cactus-parity/`](../results/raw/2026-07-10-cactus-parity/) にある。

## 要約

1. **公表値は再現する。** iPhone 17 Pro で prefill 704.8 tok/s / decode 36.3 tok/s / peak 651 MB。
   公表値 729 / 37 / 644 に対して 3% 以内。
2. **prefill の指標は誠実である。** `prefill_tps` は確かに KV キャッシュ移送分を分母から引いて
   いるが、その項は実測 0.01〜0.03 ms でしかない。実時間の `promptTokens / TTFT` と 0.01% 未満
   しか違わない。
3. **同じ端末・同じモデルで LiteRT-LM は prefill 約 5 倍、decode 約 1.6 倍速い。**
4. **速度差より品質差のほうが大きい。** 出力形式を指定した GSM8K で、Cactus の CQ4 版
   Gemma-4-E2B は 100 問すべてにおいて要求された `#### <number>` 行を一度も出さず、正答率 3.0%。
   LiteRT-LM の int4-QAT 版は 98 問で出し、正答率 88%（非量子化アンカーは 92%）。

したがって本題は速度比較ではない。多段推論を失った 4bit フォーマットが速いのは、間違えるのが
安上がりなのと同じ理由による。

なお本リポジトリ自身のハーネスで見つけたものが 3 つある。それぞれ独立に知る価値がある。
`mlx-swift-lm` は **現行の** Gemma-4-E2B 変換（2026-07-06 に KV 共有付きで再アップロード）を
ロードできない。llama.cpp アダプタはラン間で KV キャッシュをクリアしておらず、タスクの
サンプリング指定も無視していた。そして `totalGenerationTimeSeconds` は後処理の 200ms スリープを
生成時間に数えていた。

## 方法

Cactus の LLM ベンチ (`cactus-engine/tests/test_benchmark.cpp`) は、プロンプトを構築して
**ちょうど 1000 トークン**に切り詰め、**ちょうど 100 トークン**をデコードし、各ランの前に KV
キャッシュをリセットした上で、**ウォームアップ 1 回のあと 3 回の平均**を報告する。本リポジトリの
`CactusParityTask` はこれを写している。1022 トークンのプロンプト（lorem ブロック 17 個 — 彼らの
1000 と同じ 8×128 の prefill チャンク枠に収まるので、どちらもチャンク 1 個分を余計に払わない）、
100 トークン予算、greedy、`--runs 4` 駆動でラン 1 がコールド、ラン 2〜4 がウォーム。

指標は両者が同じ測り方になるよう定義した。

- **prefill tok/s** = `promptTokens / TTFT`。ここに登場するどのエンジンでも算出でき、ユーザーが
  実際に待たされる量でもある。Cactus はこれを `ttft_prompt_tps` として出力している。
- **decode tok/s** = TTFT を除いた各エンジン自身の定常デコード率。Cactus は
  `(n-1)/(total-TTFT)`、本リポジトリは `n/decodeTime`。n=100 では定義差は約 1%。
- **peak MB** = `phys_footprint`。両ハーネスが読む `task_info(TASK_VM_INFO)` の同じ値。

`scripts/cactus_parity_report.py` が以下のすべての表を生データから再生成する。

### `prefill_tps` について

`cactus-engine/src/complete.cpp:1503` を読むと、公表の prefill が水増しされているように見える。

```cpp
cache_prime_compute_ms = max(0.0, cache_prime_ms - cache_state_copy_ms);
prefill_tps            = cache_prime_tokens * 1000.0 / cache_prime_compute_ms;
```

`cache_state_copy_ms` は `move_cache_states()` — Cactus が prefill グラフと decode グラフを分けて
いる設計に固有の KV 移送 — の実測時間であり、公表レートから除外されている。**しかし実測すると
無害である。** この移送は Mac の 365 ms、iPhone の 1420 ms の prefill に対して 0.01〜0.03 ms しか
かからない。ゆえに `prefill_tps` と誠実な `ttft_prompt_tps` は 0.01% 未満で一致する。レポート
スクリプトは毎回これを検算し、信用に頼らない。

| ラン | 公表 `prefill_tps` | `ttft_prompt_tps` | `cache_state_copy_ms` | 乖離 |
|---|---:|---:|---:|---:|
| iPhone 17 Pro, auto | 704.8 | 704.8 | 0.02 | 0.004% |
| iPhone 17 Pro, metal | 710.8 | 710.8 | 0.01 | 0.001% |
| iPhone 17 Pro, cpu | 220.8 | 220.8 | 0.02 | 0.000% |
| M4 Max, auto | 2738.4 | 2738.2 | 0.01 | 0.008% |

## 速度 — iPhone 17 Pro (A19 Pro)、Gemma-4-E2B、ウォーム 3 回平均

| エンジン | 量子化 | prefill tok/s | decode tok/s | TTFT ms | peak MB |
|---|---|---:|---:|---:|---:|
| **LiteRT-LM**（自身の `benchmark()` API、1000 tok） | int4 QAT | **3893.7** | 56.9 | 275.8 | — |
| **LiteRT-LM**（本リポジトリのタスク、1022 tok） | int4 QAT | **3190.5** | **57.3** | 320.3 | 688 |
| llama.cpp | Q4_K_M | 1989.9 † | n/a ‡ | — | 278 |
| Cactus (`--backend metal`) | CQ4 | 710.8 | 36.4 | 1408.7 | 648.8 |
| Cactus (`--backend auto`) | CQ4 | 704.8 | 36.3 | 1420.1 | 651.1 |
| Cactus (`--backend cpu`) | CQ4 | 220.8 | 15.0 | 4538.9 | 552.2 |
| CoreML-LLM | INT4 palettized | ≈28 § | 26.4 | 36472 | 1704 |
| MLX-Swift | Q4 | — ¶ | — ¶ | — | — |
| _Cactus 公表値_ | CQ4 | _729_ | _37_ | — | _644_ |

† llama.cpp 自身の prefill 窓。このプロンプトでは 0 トークンしか生成しないので TTFT が無い。
‡ llama.cpp は short-chat の 12 トークンより長いプロンプトで、最初のトークンとして即 EOG を
greedy に選ぶ（543 トークンのプロンプトでも同様。short-chat では 37.5 tok/s で 128 トークン生成）。
unsloth の Gemma-4 GGUF に対するチャットテンプレート/特殊トークンの不整合が疑わしい。
§ CoreML はプロンプトトークン数を報告しない。実測 TTFT から `1022 / 36.5 秒`。
¶ `mlx-swift-lm` は **現行の** `mlx-community/gemma-4-e2b-it-4bit` をロードできない。同リポジトリは
2026-07-06 に再アップロードされ（"Re-upload MLX conversion from google/gemma-4-E2B-it@70af34e2"）、
旧リビジョン (`2c3e507`、2026-05-19) は全 35 層に `k_proj`/`v_proj` を持つ（2649 テンソル）が、
現行は 0〜15 層のみに持ち、16 層目以降は KV を共有する（2511 テンソル）。`mlx-swift-lm` の Gemma4
アテンションは全層で `k_proj`/`v_proj` を構築し、KV 共有層をスキップしないため、新レイアウトでは
15 層目でキー欠落エラーになる。アプリが解決した `main` (`b95dc78`)、`Package.resolved` が固定する
リビジョン (`5b7e543`)、macOS CLI のいずれでも再現。本リポジトリの以前の iPhone MLX 値（decode
46.2 tok/s、2026-06-17）は旧リビジョンでの計測である。Python の `mlx-lm` は現行リポジトリを問題なく
読み、下の品質ゲートで採点しているのはそちらである。

独立した 2 つの LiteRT 測定が一致している。自身の `benchmark(prefillTokens:decodeTokens:)`
（`cactus_benchmark_tokens` の直接の対応物）と、実プロンプト 1022 トークンを通すアプリ経路である。
TTFT も prefill レートと整合する（`1000/4171 = 240 ms` + デコード 1 ステップ ≈ 実測 257 ms）。

`auto` ≒ `metal` であり、実際 `cactus_backend_select("auto")` は `-1` を返して
`cactus_default_backend()` に落ち、Metal が使えればそれを選ぶ。つまり公表行は **Metal GPU** の
数字である。どちらのエンジンも ANE を使っていない。

CoreML-LLM は逆方向の外れ値である。1K トークンのプロンプトに TTFT **30〜41 秒**を要し（しかも
端末が熱くなるにつれてラン毎に増える）、LiteRT の 0.32 秒、Cactus の 1.4 秒と比べものにならない。

### このプロトコルはスマホで定常状態に達しない

Cactus のウォーム 3 ランは約 40 秒で終わり、その間 prefill は単調に落ちる。

| | ウォームアップ | ラン 1 | ラン 2 | ラン 3 |
|---|---:|---:|---:|---:|
| prefill tok/s | 764.3 | 733.8 | 698.1 | 682.7 |
| TTFT ms | 1308.5 | 1362.9 | 1432.6 | 1464.8 |

公表の 729 tok/s は事実上 1 本目の測定ランである。M4 Max では同じプロトコルで減衰が見られない
（2719.7 → 2745.0 → 2750.4）。40 秒で取ったスマホの数字はバースト値であって持続値ではない。

### peak memory は `phys_footprint` なので mmap した重みが見えない

両ハーネスとも `task_info(TASK_VM_INFO).phys_footprint` — jetsam が見る値 — を読む。クリーンな
file-backed ページは計上されないので、展開 3.8 GB のバンドルが iPhone で 651 MB、Mac で 1433 MB
と報告される。jetsam の余裕を測る指標としては正しく、モデルサイズの表明ではない。

## 品質 — GSM8K、0-shot CoT、greedy、同一プロンプト・同一抽出

エンジン間の速度は出力品質が揃って初めて意味を持つ。そして各エンジンは自前の 4bit フォーマットを
積んでいる。Cactus CQ4 は Hadamard 回転 + グループごと codebook の PTQ、LiteRT-LM は int4 QAT、
MLX は affine 4bit グループ量子化である。同じ 100 問、同じプロンプト（末尾に `#### <number>` 行を
要求する）、同じ抽出、`max_tokens=1024`。

| ビルド | 量子化 | GSM8K | `#### <n>` に未到達 | 平均出力文字数 | n |
|---|---|---:|---:|---:|---:|
| _bf16（非量子化アンカー、CPU）_ | _なし_ | _92.0%_ | _1 / 100_ | _1290_ | _100_ |
| LiteRT-LM `.litertlm` | int4 QAT | **88.0%** | 2 / 100 | 1287 | 100 |
| MLX 4bit | affine PTQ | 78.0% | 25 / 100 | 2527 | 100 |
| **Cactus CQ4** | 回転 + codebook PTQ | **3.0%** | **100 / 100** | 4241 | 100 |

アンカーは天井を 92% に置く。LiteRT の int4-QAT はその 4 ポイント下、汎用の 4bit PTQ (MLX) は
14 ポイント下、CQ4 はそもそも生き残らない。この順序 — **QAT はほぼパリティ、汎用 PTQ は劣化** —
は本リポジトリが Falcon3 と Qwen3 で得たのと同じ結論であり、今回は Gemma-4 で非量子化の基準点付きで
確認された。

MLX の 78% は下限である — 25 問は予算切れ時点でまだ推論中だった。LiteRT の 88% はほぼ確定値。
Cactus の 3.0% は、モデルが解答行に到達しないときに抽出の「最後の数値」フォールバックが拾う値で
あって、推論スコアではない。

破棄した測定が 3 つあり、いずれも安価な対照実験ひとつで捕まえた。

- `max_tokens=640` では 4bit 3 者が 2% / 76% / 33% だった。その予算では打ち切りと誤答が区別できない。
  上表が正答率と並べてマーカー到達率を報告しているのはそのためである。
- bf16 アンカーを最初 MPS で走らせたところ、50 問中 18 問でトークンの縮退（`**\n\n**\n\n**`、
  `}\n}\n}`）を起こして 64% となった。あれは Metal の数値バグを測っているのであってモデルではない。
  `results/raw/2026-07-10-cactus-parity/rejected/` に隔離してある。
- そのアンカーを n=25 で走らせた一方、他は n=100 だった。異なる問題集合を比べており、LiteRT の 88% が
  84% の天井を上回っているように見えていた。同じ 25 問ではアンカー 84% / LiteRT 80%。n=100 で取り直すと
  アンカーは 92% で、量子化版が下に来るという当然の順序になる。

### これは測定側ではなくモデル側の問題である

`scripts/cactus_cq4_ablation.py` が、代替説明を順に潰す。

- **チャットテンプレート。** `cactus_render_prompt` は
  `<bos><|turn>user\nHi<turn|>\n<|turn>model\n` を返す — Gemma-4 の HuggingFace リファレンス
  テンプレートとバイト単位で同一。
- **stop sequences。** Gemma-4 のターン終端は `<end_of_turn>` ではなく `<turn|>` である。
  `<end_of_turn>` 指定、`<turn|>` 指定、指定なしの 3 通りとも、同じ 3851 文字を出力する。
- **thinking モード。** Cactus 自身のベンチはシステムプロンプトを `/no_think` で始める。
  `enable_thinking_if_supported=False`、そのシステムプロンプトの付与、その両方 — いずれも
  何も変えない。
- **トークン予算。** 1536 トークンでも 10 問中 10 問がマーカーに到達せず、平均 6656 文字。
  余地が足りないのではない。

これらが消えると、破綻箇所が特定できる。CQ4 は `What is 17 + 25?` に 40 トークンで正答して終了する。
1 段階の文章題（`A box has 48 apples. Half are red.`）にも正答する — ただし *「もし『半分』が
きっかり半分を意味するなら」* と留保を付ける。2 段階の問題を与えると、そもそも読みを確定しない。

> Natalia sold clips to 48 friends in April. This implies that she sold $N_{April} = 48$ clips
> (assuming the problem implies a direct relationship, or we assume the number of friends is
> the quantity). Let's assume Natalia sold $X$ clips in April. … Let's re-evaluate the structure.

そして予算が尽きるまで続く。LiteRT-LM の int4-QAT 版は両方の問題に直接答える。CQ4 では流暢さと
1 段階の算術は生き残るが、多段の文章題で読みを確定する能力は生き残らない。

### `####` マーカーの問題でもない

CQ4 はマーカーを半分しか出せない。「`#### 42` をそのまま複写せよ」には従うが、「30 まで数えて最後に
マーカーで締めよ」では 85 トークンで `# 30` と書き、「ハッシュ文字を 4 つ書け」では `abcd 42` と書く。
ここから当然の反論が出る — 推論はできていて `####` が打てないだけではないか。CQ4 が出せるマーカー
(`The answer is <number>.`) で同じ 50 問を採点し直すと、CQ4 は **2.0%**、LiteRT は **82.0%**
（`#### <n>` 版では同 50 問で 4.0% と 86.0%）。CQ4 は依然として平均 4237 文字まで喋って予算に当たる。
マーカーの欠陥は実在するが独立した現象であり、スコアの原因ではない。

### 彼らのハイブリッドも救いにならない

オンデバイスのモデルを測るためにクラウド handoff を切っている。handoff は Cactus の出荷時既定なので
不公平だという反論があり得るが、そうではない。CQ4 が誤答する GSM8K の問題に対し、Cactus 自身の信頼度
プローブは自らの閾値 0.81 に対して **0.9399〜0.9999** を返し、`cloud_handoff=false`「above threshold」
と報告する（6/6 で確認）。出荷時の既定は、ローカルで、自信を持って、誤答する。

これは Cactus の「CQ は ARC・MMLU・GSM8K といったベンチで品質を維持しつつ大幅な圧縮を可能に
する」という主張と食い違う。測定は、彼ら自身の `cactus benchmark` がダウンロードするキャリブ
レーション済みバンドルに対し、彼ら自身の `cactus_complete` FFI を通し、greedy サンプリングと
クラウド handoff 無効の条件で行っている。

### どの数字をどこで測ったか

GSM8K の数字はすべて Mac で、iPhone なのは速度の表だけである。Cactus についてはその隙間を実測で塞いだ。
同じ 1000 トークンのプロンプトと greedy デコードで、その Metal 経路は **iPhone 17 Pro と M4 Max で
トークン単位に完全一致する出力を出す**（完了 ID 100/100 一致。CPU 経路はトークン 6 で分岐する）。
したがって品質の結論は、ベンチした端末にそのまま転移する。LiteRT の GSM8K は Mac CPU で走らせた。
パリティ・プロンプトに対する iPhone Metal の出力は正常だが、GSM8K を実機で取り直してはいない。本
リポジトリには LiteRT が Apple GPU 上でだけ縮退した実例（gemma-4-12B、CPU 経路は正常）があるので、
これは明記しておく。MLX の品質は Python の `mlx-lm` であり、その経路は端末では動かない。

## 再現手順

```bash
# 実機上の Cactus。tests/ios/run.sh は iOS 26+ で起動するために 2 箇所の修正が要る
# （UIScene デリゲートと IPHONEOS_DEPLOYMENT_TARGET ≥ 15.0）。下記参照。
cactus benchmark --ios

# 本リポジトリ、条件を揃えた測定
scripts/bench_cactus_parity_iphone.sh all
python3 scripts/cactus_parity_report.py

# 品質ゲート
python3 scripts/gsm8k_cactus_vs_litert.py --backend litertlm --path <…>.litertlm --n 100 --max-tokens 1024
python3 scripts/gsm8k_cactus_vs_litert.py --backend cactus --cactus-repo <clone> --path <clone>/weights/gemma-4-e2b-it-cq4 --n 100 --max-tokens 1024
python3 scripts/cactus_cq4_ablation.py --cactus-repo <clone> --path <clone>/weights/gemma-4-e2b-it-cq4
```

現行のツールチェーンでは `cactus benchmark --ios` が 2 箇所で止まる。いずれもエンジンとは無関係
である。テストアプリが `UIApplicationDelegate` のみで書かれているため、iOS 26+ は起動時に
`__UIApplicationEvaluateRuntimeIssueForNoSceneLifecycleAdoption` でトラップする
（`EXC_BREAKPOINT`）。シーンマニフェスト *と* 実際の `UIWindowSceneDelegate` の両方が要る。
また `tests/ios/run.sh` は `IPHONEOS_DEPLOYMENT_TARGET=13.0` を設定するが、これは Xcode 27 の
下限 15.0 を下回る。

## 留保

- Cactus の Mac 行は M4 **Pro** であり、ここで使った Mac は M4 **Max**（prefill 2738 tok/s /
  decode 141.6 / 1433 MB）。比較できないので Mac の直接対決は行っていない。
- ~~macOS の LiteRT-LM は CPU のみで動く（GPU 経路が OpenCL で、Apple silicon では死んでいる）~~
  **2026-07-13 撤回:** これは良性のログ行の誤読だった。`Failed to create OpenCL context` は
  *正常に動く* GPU 経路でも出力され、直後に `Created Metal device from provided device id` が続く。
  macOS の LiteRT-LM は WebGPU/Dawn→Metal 経由で GPU 上で動作し、warm では公式 E2B カード値を
  ~5% 以内で再現する（decode ~152 vs 160、prefill ~7.5k vs 7,835。ログは
  [`results/raw/2026-07-13-e2b-mac-webgpu/`](../results/raw/2026-07-13-e2b-mac-webgpu/)）。
  iPhone 17 Pro が Cactus の公表行であることは変わらず、それがここでの直接対決の土俵である。
- このプロンプトで LiteRT は約 32 トークンで EOS に達するが、Cactus は 100 トークンまで強制される。
  この KV 深度（1022 → 1055 対 1122）ではデコード率はトークン数にほぼ依存しないが、窓は同一ではない。
- GSM8K は全 1319 問ではなく、テストセット先頭 100 問で採点した。
