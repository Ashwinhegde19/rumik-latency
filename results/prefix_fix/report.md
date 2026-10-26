# Prefix experiment: results

GPU NVIDIA H100 80GB HBM3; torch 2.8.0+cu128; model revision 0ed3c986; silence threshold -40.0 dBFS per 80 ms frame. 320 clips. Determinism (k0 loop vs StreamingTTS.run): {'job': 'hi_en_mixed_medium', 'seed': 1000, 'tokens_equal': True, 'n_tokens': [576, 576], 'first_nonsilent_frame': [2, 2]}. k0 reproduces the audit's optimized tokens: {'n': 80, 'equal': 80}. Errors: 0.

Prefixes (real generated starts; codebook-0 code per frame):
- A from `hinglish_medium_seed1009`: [1049, 127, 1880]
- B from `hi_medium_seed1001`: [1049, 127, 1880]
- A4 from `hinglish_medium_seed1001`: [1049, 127, 1880, 972]

## Where does the model start speaking?

| cond | prefix frames | n | first non-silent generated frame: 0 / 1 / 2 / 3 / 4+ / none | median | first generated code (codebook 0) | ttfa to first audible, p50 ms | stop_predictor |
|---|---:|---:|---|---:|---|---:|---:|
| k0 | 0 | 80 | 0 / 1 / 16 / 51 / 12 / 0 | 3.0 | 1049 (frame-0 transient) ×80 | 110 | 79/80 |
| k1 | 1 | 40 | 0 / 8 / 32 / 0 / 0 / 0 | 2.0 | 127 (frame-1 transient) ×40 | 84 | 40/40 |
| k2 | 2 | 40 | 17 / 22 / 1 / 0 / 0 / 0 | 1.0 | 1766 ×20; 1880 (frame-2 transient) ×8; 1156 ×5 | 59 | 40/40 |
| k3 | 3 | 80 | 62 / 16 / 2 / 0 / 0 / 0 | 0.0 | 1766 ×37; 972 ×11; 1272 ×11 | 34 | 80/80 |
| k4 | 4 | 40 | 3 / 4 / 2 / 0 / 31 / 0 | 9.0 | 972 ×36; 1272 ×2; 1766 ×2 | 268 | 40/40 |
| k3B | 3 | 40 | 25 / 13 / 1 / 0 / 1 / 0 | 0.0 | 1766 ×16; 972 ×12; 1272 ×5 | 34 | 40/40 |

## Quality of the generated audio (means; latency rows are medians)

| metric | k0 | k1 | k2 | k3 | k4 | k3B |
|---|---:|---:|---:|---:|---:|---:|
| first_nonsilent_frame | 3.00 | 2.00 | 1.00 | 0.00 | 9.00 | 0.00 |
| ttfa_nonsilent_ms | 110 | 84 | 59 | 34 | 268 | 34 |
| ttfa_first_frame_ms | 32 | 32 | 33 | 33 | 34 | 33 |
| frames | 92.7 | 93.3 | 89.7 | 91.1 | 102.2 | 89.5 |
| cer_whisper | 0.017 | 0.016 | 0.030 | 0.017 | 0.022 | 0.029 |
| cer_mms | 0.059 | 0.063 | 0.084 | 0.056 | 0.066 | 0.067 |
| utmos22 | 3.577 | 3.565 | 3.450 | 3.563 | 3.685 | 3.464 |
| utmosv2 | 2.850 | 2.767 | 2.684 | 2.799 | 2.806 | 2.681 |
| spk_ecapa | 0.8610 | 0.8608 | 0.8604 | 0.8647 | 0.8713 | 0.8449 |
| spk_wavlm | 0.9826 | 0.9828 | 0.9833 | 0.9824 | 0.9847 | 0.9780 |

## Paired differences versus k0 (same text and seed; stratified bootstrap 95% CI, sign-flip p)

| metric | k1 − k0 | k2 − k0 | k3 − k0 | k4 − k0 | k3B − k0 |
|---|---:|---:|---:|---:|---:|
| first_nonsilent_frame | -1.73 [-2.50, -1.05] p=0.000 **distinguishable** | -2.92 [-3.67, -2.27] p=0.000 **distinguishable** | -2.99 [-3.42, -2.61] p=0.000 **distinguishable** | +3.77 [+2.30, +5.22] p=0.000 **distinguishable** | -3.05 [-3.88, -2.30] p=0.000 **distinguishable** |
| ttfa_nonsilent_ms | -44 [-64, -27] p=0.000 **distinguishable** | -75 [-95, -57] p=0.000 **distinguishable** | -76 [-88, -66] p=0.000 **distinguishable** | +101 [+63, +139] p=0.000 **distinguishable** | -78 [-100, -58] p=0.000 **distinguishable** |
| ttfa_first_frame_ms | +1 [+0, +1] p=0.000 **distinguishable** | +1 [+1, +1] p=0.000 **distinguishable** | +2 [+2, +2] p=0.000 **distinguishable** | +3 [+2, +3] p=0.000 **distinguishable** | +2 [+1, +2] p=0.000 **distinguishable** |
| frames | -0.0 [-3.8, +3.3] p=1.000 | -3.6 [-7.0, -0.5] p=0.055 | -1.6 [-4.0, +0.8] p=0.221 | +8.9 [+4.5, +13.1] p=0.001 **distinguishable** | -3.8 [-7.1, -0.7] p=0.037 **distinguishable** |
| cer_whisper | -0.003 [-0.009, +0.003] p=0.477 | +0.010 [+0.001, +0.020] p=0.049 | -0.000 [-0.004, +0.003] p=0.825 | +0.003 [-0.003, +0.009] p=0.411 | +0.010 [-0.000, +0.021] p=0.142 |
| cer_mms | -0.002 [-0.029, +0.022] p=0.907 | +0.018 [-0.018, +0.052] p=0.444 | -0.003 [-0.024, +0.015] p=0.776 | +0.000 [-0.038, +0.034] p=0.996 | +0.001 [-0.025, +0.027] p=0.940 |
| utmos22 | -0.058 [-0.160, +0.042] p=0.322 | -0.174 [-0.280, -0.076] p=0.004 **distinguishable** | -0.014 [-0.106, +0.079] p=0.766 | +0.062 [-0.048, +0.177] p=0.344 | -0.159 [-0.270, -0.045] p=0.025 **distinguishable** |
| utmosv2 | -0.071 [-0.171, +0.028] p=0.281 | -0.154 [-0.290, -0.007] p=0.084 | -0.051 [-0.136, +0.034] p=0.271 | -0.032 [-0.161, +0.101] p=0.667 | -0.157 [-0.300, -0.019] p=0.077 |
| spk_ecapa | -0.0047 [-0.0157, +0.0062] p=0.462 | -0.0051 [-0.0153, +0.0049] p=0.444 | +0.0038 [-0.0032, +0.0109] p=0.339 | +0.0058 [-0.0036, +0.0148] p=0.298 | -0.0206 [-0.0325, -0.0085] p=0.012 **distinguishable** |
| spk_wavlm | +0.0009 [-0.0048, +0.0062] p=0.767 | +0.0015 [-0.0015, +0.0045] p=0.384 | -0.0002 [-0.0032, +0.0028] p=0.885 | +0.0029 [-0.0006, +0.0064] p=0.212 | -0.0039 [-0.0068, -0.0006] p=0.082 |

## CER by language group (Whisper; mean, k0 vs k3)

| group | n | k0 | k3 | k3 − k0 CI95 |
|---|---:|---:|---:|---|
| en | 24 | 0.000 | 0.000 | [+0.000, +0.000] |
| hinglish | 24 | 0.027 | 0.028 | [-0.008, +0.009] |
| hi | 24 | 0.029 | 0.025 | [-0.013, +0.005] |
| hi_en_mixed | 8 | 0.003 | 0.007 | [-0.005, +0.013] |

## Read

Numbers only; the interpretation key was written before the run (notes.md, 24 Oct). k0 is the optimized path as audited; the prefix frames are never part of the output audio; ttfa is measured from request start and includes prefix decoding and the longer prefill.

