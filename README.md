# SeasonAudio

SeasonAudio for Stable Audio 3 Models is a .NET audio generation library for ONNX bundles converted from the Stable Audio 3 Python project.

The initial public release focuses on a simple application-facing API:

- instance-based text-to-audio generation through `new StableAudio(model)` and `Generate(...)`
- built-in model alias resolution and path-based bundle loading
- standard WAV byte output suitable for saving directly as `.wav`
- optional debug diagnostics controlled by `StableAudio.EnableDebugOutput`
- CPU-first integration with optional execution provider selection

The package name is intentionally broader than `Stable Audio` because the long-term goal is to support more audio model families over time. The current implementation focuses on Stable Audio 3 compatible ONNX workflows, but the project is not an official Stability AI project and does not ship Stability AI model weights.

## Naming

- Package name: `SeasonAudio`
- Main runtime API: `Season.AI.StableAudio`
- Reason: the package name stays neutral and future-proof, while the current API name remains explicit about the model family it supports today

## Project Links

- GitHub: [SeasonRealms/SeasonAudio](https://github.com/SeasonRealms/SeasonAudio)
- Models: https://huggingface.co/SeasonEngine/stable-audio-3-onnx
- ONNX export workflow: [export-stable-audio-onnx.yml](https://github.com/SeasonRealms/SeasonAudio/actions/workflows/export-stable-audio-onnx.yml)

If you prefer to build the ONNX models yourself, clone the repository and run the workflow logic locally.

## Origin

- The audio pipeline is translated and adapted from [stable-audio-3](https://github.com/Stability-AI/stable-audio-3).
- The intended model inputs are ONNX files converted from stable-audio-3 Python models.
- Files translated from stable-audio-3 Python sources include source attribution and an explicit modification notice in the file header.

## Features

- Stable Audio 3 text-to-audio inference from .NET
- Supports built-in aliases and direct model bundle paths
- Returns standard RIFF/WAVE bytes instead of requiring a specific playback runtime
- Prompt guidance with optional negative prompt
- Deterministic generation through `seed`
- Optional provider selection: `auto`, `cpu`, `dml`, `cuda`, `coreml`, `nnapi`
- Constructor-side model/session initialization with a leaner inference hot path
- Internal zero-condition caching and reduced temporary buffer allocation during sampling
- Optional debug logging through `StableAudio.EnableDebugOutput`
- Extra troubleshooting helpers via `StableAudio.DebugPromptEncoding(...)` and `StableAudio.DebugFirstDiTStep(...)`

## Install

```bash
dotnet add package SeasonAudio
```

## Supported Models

`SeasonAudio` currently supports these Stable Audio 3 model ids and aliases:

| Model family | Hugging Face model id | Accepted `model` values | Typical use |
|---|---|---|---|
| Small SFX | `stabilityai/stable-audio-3-small-sfx` | `stable-audio-3-small-sfx`, `small-sfx`, `small_sfx`, `sa3-sm-sfx` | short sound effects, one-shots, UI sounds |
| Small Music | `stabilityai/stable-audio-3-small-music` | `stable-audio-3-small-music`, `small-music`, `small_music`, `sa3-sm-music` | lightweight music and loops |
| Medium | `stabilityai/stable-audio-3-medium` | `stable-audio-3-medium`, `medium`, `sa3-m` | higher-capacity general generation |

You can also pass:

- a model bundle directory that contains `dit.onnx`
- the full path to a `dit.onnx` file

Bundle layout is expected to resolve the matching decoder and `t5gemma` tokenizer/text-encoder files alongside the model root.

Model notes:

- `stable-audio-3-small-sfx` is best suited for short clips. For many SFX cases, `3-7` seconds with `durationPaddingSeconds` around `0-1` gives more stable output.
- `stable-audio-3-small-music` is the lightweight choice when you want faster iteration.
- `stable-audio-3-medium` is the higher-capacity option when quality matters more than footprint.

## Quick Start

```csharp
using Season.AI;

StableAudio.EnableDebugOutput = true; // optional

var stableAudio = new StableAudio(
    model: "stable-audio-3-small-sfx",
    provider: "cpu");

byte[] wavBytes = stableAudio.Generate(
    prompt: "A short sci-fi UI confirmation beep with a soft digital tail",
    seconds: 4f,
    steps: 8,
    cfgScale: 1.5f,
    seed: 12345);

File.WriteAllBytes(@"output.wav", wavBytes);
```

Because the caller environment may not have a predictable audio playback stack, the recommended integration pattern is to save the returned bytes as a `.wav` file and let the application decide how to play or distribute it.

Example using the music variant:

```csharp
using Season.AI;

var stableAudio = new StableAudio("stable-audio-3-small-music");

byte[] wavBytes = stableAudio.Generate(
    prompt: "Warm ambient synth loop with soft pads and gentle arpeggio",
    negativePrompt: "harsh noise, clipping, distorted vocal",
    seconds: 12f,
    steps: 8,
    cfgScale: 2.0f);

File.WriteAllBytes(@"music-preview.wav", wavBytes);
```

## Why This API

This initial API is intentionally simple:

- the caller creates a `StableAudio` instance with a model alias or a prepared ONNX bundle path
- the constructor resolves the required DiT, decoder, text encoder, and tokenizer assets
- the instance `Generate(...)` call returns ready-to-save WAV bytes
- model loading stays outside the hot inference loop, which helps repeated generation scenarios
- the host application remains free to choose its own file, playback, transport, or storage flow

For the first release, the recommended runtime target is CPU execution unless you already control the ONNX Runtime provider environment on the target machine.

## API Notes

`StableAudio` current initialization and generation shape:

```csharp

public StableAudio(string model, string? provider)

public byte[] Generate(
    string prompt,
    float seconds = 10f,
    int steps = 8,
    float cfgScale = 1f,
    int? seed = null,
    string? negativePrompt = null,
    float sigmaMax = 1f,
    float apgScale = 1f,
    float decoderScale = 1f,
    float durationPaddingSeconds = 6f,
    bool truncateOutputToDuration = true)
```

Parameter behavior:

- `model`: constructor parameter; accepts model alias, model bundle directory, or full `dit.onnx` path
- `provider`: optional constructor parameter; selects the ONNX Runtime provider during initialization, with supported values `auto`, `cpu`, `dml`, `cuda`, `coreml`, `nnapi`
- `prompt`: positive text prompt used for audio generation
- `seconds`: requested output duration in seconds; must be greater than `0`
- `steps`: denoising step count; must be greater than `0`
- `cfgScale`: classifier-free guidance strength; `1` means no extra guidance branch
- `seed`: optional random seed for reproducible output
- `negativePrompt`: optional negative prompt used when `cfgScale` is not `1`
- `sigmaMax`: sampler noise ceiling; valid range is `0.01` to `1.0`
- `apgScale`: APG guidance factor; valid range is `0` to `1`
- `decoderScale`: latent scaling before decoder; must be greater than `0`
- `durationPaddingSeconds`: extra latent duration added before decoding; useful when a model benefits from a little headroom
- `truncateOutputToDuration`: when `true`, trims the final WAV to `seconds`; when `false`, keeps the padded duration

Practical tuning guidance:

- increase `steps` when you want more refinement at the cost of speed
- increase `cfgScale` when prompt adherence is too weak
- set `seed` when you need deterministic results
- reduce `durationPaddingSeconds` for short SFX generation if extra tail content becomes undesirable
- keep `truncateOutputToDuration = true` for most application-facing output files

## Debug Output

Enable debug output before calling the generation API if you need internal sampling diagnostics:

```csharp
StableAudio.EnableDebugOutput = true;
```

When enabled:

- internal diagnostics are emitted through `Debug.WriteLine(...)`
- provider choice and fallback plan are logged
- prompt encoding statistics are logged
- per-step sampling and decoder statistics are logged

When disabled:

- debug logging is skipped
- generation still works normally

The default value is:

```csharp
StableAudio.EnableDebugOutput = false;
```

Optional troubleshooting helpers:

- `StableAudio.DebugPromptEncoding(...)` returns a text report for tokenizer ids, attention mask, and prompt hidden-state previews
- `StableAudio.DebugFirstDiTStep(...)` returns a deeper text report for the first DiT sampling step

These helpers are useful for integration validation and prompt/tokenizer diagnostics, but they are not required for normal generation.

## Result Format

`StableAudio.Generate(...)` returns a `byte[]` containing a standard `.wav` file:

- sample format: 16-bit PCM WAV
- sample rate: `44100`
- output is ready to save with `File.WriteAllBytes(...)`

This makes the API easy to use in desktop apps, tools, game pipelines, background jobs, and services where the actual playback environment may differ across clients.

## Models

Recommended model sources:

- export the ONNX bundle yourself from an authorized Stability AI model checkout
- keep the generated bundle in a location that the application can resolve, such as `Models/...`

The runtime expects a bundle composed of:

- DiT model
- SAME decoder
- T5Gemma text encoder
- tokenizer assets

## License Split

- This repository code is released under the MIT License.
- Stable Audio 3 model weights are not covered by this repository MIT license.
- Stable Audio 3 model access and use are governed separately by Stability AI terms and the model-specific license flow:
  - `https://stability.ai/license`
  - `https://huggingface.co/collections/stabilityai/stable-audio-3`
  - the gated Hugging Face model pages published by Stability AI

## Model Access Requirements

To export or run your own Stable Audio 3 ONNX bundle, the caller must provide their own Hugging Face credentials and must already have permission to download the source model.

Typical access flow:

1. Create or sign in to your Hugging Face account.
2. Visit the gated model page for the variant you want to use.
3. Review and accept the applicable Stability AI terms.
4. Complete any access request form required by Stability AI or Hugging Face.
5. Wait until the request is approved if the repository is not immediately available.
6. Create a Hugging Face access token with permission to read gated model repositories.
7. Export `HF_TOKEN` before running the export script or GitHub workflow.

Example:

```bash
export HF_TOKEN=hf_your_token_here
```

PowerShell:

```powershell
$env:HF_TOKEN = "hf_your_token_here"
```

If your account does not have access to the selected gated repository, the export script will fail even when `HF_TOKEN` is present.

## Export Stable Audio 3 To ONNX

The repository includes `scripts/export_stable_audio_onnx.py` for exporting:

- the DiT model
- the paired SAME decoder
- the T5Gemma text encoder
- the tokenizer bundle

Quick examples:

```bash
python scripts/export_stable_audio_onnx.py --variant small-sfx --out-dir out
python scripts/export_stable_audio_onnx.py --variant small-music --out-dir out
python scripts/export_stable_audio_onnx.py --variant medium --out-dir out
```

The script accepts optional advanced overrides, but `--variant` is the recommended entry point.

## ONNX External Data And Runtime Compatibility

Models larger than the 2 GiB protobuf limit (the `medium` DiT and its SAME decoder) are written by `torch.onnx.export` as one external-data shard per tensor. Both `scripts/export_stable_audio_onnx.py` and `scripts/merge_stable_audio_onnx.py` consolidate those shards into a single `<graph>.onnx_data` companion per model, which is the layout the runtime bundle resolver expects.

Consolidation must keep the small `Constant` tensors that ONNX shape inference reads (for example `Unsqueeze`/`Reshape` axes) inline in the graph file. ONNX Runtime only auto-loads external data for graph *initializers*, never for *Constant-node attribute* tensors, so a file-backed constant makes session creation fail on ONNX Runtime 1.23 and later with:

```
[ShapeInferenceError] Cannot parse data from external tensors.
Please load external data into raw data for tensor: /Constant_output_0
```

Both scripts therefore save with `convert_attribute=False` and `size_threshold=1024`: only large initializers (weights) go to the `.onnx_data` file, while every constant stays inline. This keeps the bundle loadable across ONNX Runtime versions and platforms. `export_stable_audio_onnx.py` additionally verifies each consolidated model and fails the export if any constant is still external.

Use `scripts/merge_stable_audio_onnx.py` to (re)merge an already-exported bundle without re-running the PyTorch export:

```bash
# Non-destructive: write a merged copy next to the source bundle (<bundle>-merged).
python scripts/merge_stable_audio_onnx.py /path/to/stable-audio-3-medium-onnx

# Merge into an explicit directory, or consolidate the bundle in place.
python scripts/merge_stable_audio_onnx.py /path/to/bundle --out-dir /path/to/merged
python scripts/merge_stable_audio_onnx.py /path/to/bundle --in-place

# Verify existing model(s) without merging; --deep-verify also runs shape inference.
python scripts/merge_stable_audio_onnx.py --verify-only /path/to/dit.onnx --deep-verify
```

Requirements: `pip install "onnx>=1.16"`. Consolidation loads the full model into memory, so merging the `medium` DiT needs roughly 6-8 GB of free RAM. A model passes verification when it has zero external constant attributes, no small external initializers, a graph below 2 GiB, and (with `--deep-verify`) shape inference succeeds without loading external data.

## GitHub Workflow

The repository also includes a GitHub Actions workflow:

- `.github/workflows/export-stable-audio-onnx.yml`

The workflow exposes the same three variants and expects `HF_TOKEN` to be configured as a repository secret.

## Hugging Face Distribution Guidance

- Do not assume this MIT repository grants redistribution rights for gated weights.
- Do not publish exported ONNX bundles unless your access terms explicitly allow redistribution.
- Prefer documenting the export process and asking users to export from their own authorized Hugging Face accounts.
- If redistribution rights are unclear, the safest option is to publish only the code, scripts, and model card documentation, and not upload the converted ONNX files themselves.
- A model card note alone is not enough if the upload still contains converted weights derived from a gated repository. The right to redistribute the files must already exist before uploading them.
- In practice, for Stable Audio 3 gated models, the conservative recommendation is: do not provide direct downloadable ONNX weights unless Stability AI terms or explicit written permission clearly allow redistribution of converted derivatives.
- A safer Hugging Face pattern is:
  - repository contains README, export instructions, and optional config metadata only
  - no ONNX weight binaries are uploaded
  - users bring their own approved `HF_TOKEN` and export locally

## Disclaimer

SeasonAudio is an independent open source project. It is not affiliated with, endorsed by, or distributed by Stability AI.
