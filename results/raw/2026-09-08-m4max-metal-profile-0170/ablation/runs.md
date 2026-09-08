| run | runtime | model | tokens | steady tok/s | ms/token | dispatches | skipped | text == unmodified |
|---|---|---|--:|--:|--:|--:|--:|---|
| base | 0.17.0 | qwen3_4b_mixed_int4.litertlm | 300 | 114.2 | 8.75 |  |  | True |
| base2 | 0.17.0 | qwen3_4b_mixed_int4.litertlm | 300 | 113.8 | 8.79 |  |  | True |
| base3 | 0.17.0 | qwen3_4b_mixed_int4.litertlm | 300 | 114.5 | 8.73 |  |  | True |
| base4 | 0.17.0 | qwen3_4b_mixed_int4.litertlm | 300 | 114.4 | 8.74 |  |  | True |
| ds_i4_0131_base | 0.13.1 | model.litertlm | 300 | 114.4 | 8.74 |  |  |  |
| ds_i4_0131_skipGEMV | 0.13.1 | model.litertlm | 300 | 147.0 | 6.80 | 257894 | 17700 |  |
| ds_i4_base | 0.17.0 | model.litertlm | 259 | 125.5 | 7.97 |  |  |  |
| ds_i4_skipGEMV | 0.17.0 | model.litertlm | 300 | 261.4 | 3.83 | 261026 | 58996 |  |
| ds_q8_base | 0.17.0 | DeepSeek-R1-Distill-Qwen-1.5 | 300 | 105.8 | 9.46 |  |  |  |
| ds_q8_skipGEMV | 0.17.0 | DeepSeek-R1-Distill-Qwen-1.5 | 300 | 188.2 | 5.31 | 359871 | 42140 |  |
| i4_0131_base_r1 | 0.13.1 | model.litertlm | 300 | 138.9 | 7.20 |  |  |  |
| i4_0131_base_r2 | 0.13.1 | model.litertlm | 300 | 139.2 | 7.19 |  |  |  |
| i4_0131_skipGEMV_r1 | 0.13.1 | model.litertlm | 300 | 239.1 | 4.18 | 257894 | 58800 |  |
| i4_0131_skipGEMV_r2 | 0.13.1 | model.litertlm | 300 | 239.7 | 4.17 | 257894 | 58800 |  |
| i4_base_r1 | 0.17.0 | model.litertlm | 259 | 125.6 | 7.96 |  |  |  |
| i4_base_r2 | 0.17.0 | model.litertlm | 259 | 125.8 | 7.95 |  |  |  |
| i4_skipGEMV_r1 | 0.17.0 | model.litertlm | 300 | 266.1 | 3.76 | 261026 | 58996 |  |
| i4_skipGEMV_r2 | 0.17.0 | model.litertlm | 300 | 262.4 | 3.81 | 261026 | 58996 |  |
| q8_base_r1 | 0.17.0 | DeepSeek-R1-Distill-Qwen-1.5 | 300 | 124.2 | 8.05 |  |  |  |
| q8_base_r2 | 0.17.0 | DeepSeek-R1-Distill-Qwen-1.5 | 300 | 123.4 | 8.10 |  |  |  |
| q8_plain_r1 | 0.17.0 | DeepSeek-R1-Distill-Qwen-1.5 | 300 | 123.0 | 8.13 |  |  |  |
| q8_plain_r2 | 0.17.0 | DeepSeek-R1-Distill-Qwen-1.5 | 300 | 123.1 | 8.12 |  |  |  |
| q8_skipGEMV_r1 | 0.17.0 | DeepSeek-R1-Distill-Qwen-1.5 | 300 | 217.8 | 4.59 | 359871 | 42140 |  |
| q8_skipGEMV_r2 | 0.17.0 | DeepSeek-R1-Distill-Qwen-1.5 | 300 | 221.3 | 4.52 | 359871 | 42140 |  |
| qwen_base_r5 | 0.17.0 | qwen3_4b_mixed_int4.litertlm | 300 | 114.7 | 8.72 |  |  | True |
| qwen_skipGEMV_h | 0.17.0 | qwen3_4b_mixed_int4.litertlm | 300 | 429.3 | 2.33 | 331185 | 75852 | False |
| skipATTN | 0.17.0 | qwen3_4b_mixed_int4.litertlm | 300 | 125.1 | 7.99 | 331185 | 65016 | False |
| skipGEMV | 0.17.0 | qwen3_4b_mixed_int4.litertlm | 300 | 426.4 | 2.35 | 331185 | 75852 | False |
| skipMLP | 0.17.0 | qwen3_4b_mixed_int4.litertlm | 300 | 247.9 | 4.03 | 331185 | 43344 | False |
| skipNORM | 0.17.0 | qwen3_4b_mixed_int4.litertlm | 300 | 120.0 | 8.33 | 331185 | 43645 | False |
| skipQKV | 0.17.0 | qwen3_4b_mixed_int4.litertlm | 295 | 138.4 | 7.23 | 331185 | 32508 | False |
| skipboth | 0.17.0 | qwen3_4b_mixed_int4.litertlm | 300 | 113.7 | 8.80 | 331185 | 478 | False |
| skipdq | 0.17.0 | qwen3_4b_mixed_int4.litertlm | 300 | 113.8 | 8.79 | 331185 | 239 | False |
| skipmm | 0.17.0 | qwen3_4b_mixed_int4.litertlm | 300 | 114.0 | 8.77 | 331185 | 239 | False |
