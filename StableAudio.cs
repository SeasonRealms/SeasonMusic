// Copyright (c) SeasonEngine and contributors.
// Licensed under the MIT License.
// https://github.com/SeasonRealms/SeasonAudio
// SeasonAudio for Stable Audio Models
//
// NOTE (latentLength parity / decoder SWA reshape):
//   The dynamic decoders require `latentLength` to be EVEN. Inside the decoder the latent is
//   upsampled to the transformer resolution (seq = latentLength * upsampleFactor) and the
//   sliding-window attention (SWA) layers reshape seq into windows via `seq // windowSize`,
//   then Reshape to {1, seq/windowSize, windowSize, dim}. Because windowSize == 2 * upsampleFactor,
//   divisibility (seq % windowSize == 0) holds only when latentLength is even.
//   This applies to BOTH `same-s` (dec_dynamic_bf16.onnx, used by small-sfx / small-music) and
//   `same-l` (dec_dynamic_triton_swa.onnx, used by stable-audio-3-medium). Previously the rounding
//   was gated on UsesSameSDecoder only, so the medium (`same-l`) decoder kept an odd latentLength
//   and crashed at /decoder/layers.N/Reshape_* with:
//     "input_shape_size == requested_shape_size was false", e.g.
//     Input shape:{1,5491,768} (seconds=30 -> latentLength=323 -> seq=323*17=5491),
//     requested shape:{1,161,34,768} (5491//34=161, 161*34=5474 != 5491).
//   Fix: always round latentLength up to the next even value in Generate and DebugFirstDiTStep.
//   The extra +1 latent frame (~0.09s) is harmless; output is truncated to the requested duration.

namespace Season.Audio;

public partial class StableAudio
{
    public static bool EnableDebugOutput { get; set; } = false;

    const int SampleRate = 44_100;
    const int LatentChannels = 256;
    const int MaxTextTokens = 256;
    const int DitTrainedMinLatentLength = 256;
    const int DitMaxLatentLength = 4_096;
    const float DistributionShiftBase = 0.5f;
    const float DistributionShiftMax = 1.15f;
    const float MaxConditionSeconds = 384f;
    const int InpaintConditionChannels = LatentChannels + 1;
    const int SampleCompression = 4_096;
    static readonly int[] SingleValueDimensions = [1];
    static readonly int[] TokenMaskDimensions = [1, MaxTextTokens];
    static readonly int[] HiddenStateDimensions = [1, MaxTextTokens, 768];
    static readonly TextCondition ZeroCondition = new(
        new DenseTensor<float>(new float[MaxTextTokens * 768], HiddenStateDimensions),
        new DenseTensor<float>(new float[MaxTextTokens], TokenMaskDimensions));

    static readonly ConcurrentDictionary<string, Lazy<ModelBundle>> Bundles = new(StringComparer.OrdinalIgnoreCase);
    readonly string model;
    readonly string? provider;
    readonly ModelSpec spec;
    readonly ModelBundle bundle;

    public static string DebugPromptEncoding(
        string model,
        string prompt,
        string? provider = null,
        int previewTokenCount = 32,
        int previewHiddenTokenCount = 4,
        int previewHiddenDimensionCount = 8)
    {
        if (previewTokenCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(previewTokenCount), "The token preview count must be greater than 0.");
        if (previewHiddenTokenCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(previewHiddenTokenCount), "The hidden-token preview count must be greater than 0.");
        if (previewHiddenDimensionCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(previewHiddenDimensionCount), "The hidden-dimension preview count must be greater than 0.");

        var runtime = new StableAudio(model, provider);
        var debugInfo = runtime.EncodePromptDebug(runtime.bundle, prompt);
        var builder = new StringBuilder();
        builder.AppendLine($"model={runtime.model}");
        builder.AppendLine($"provider={DescribeProviderPlan(runtime.provider)}");
        builder.AppendLine($"normalizedPrompt={EscapeForLog(debugInfo.NormalizedPrompt)}");
        builder.AppendLine($"validTokenCount={debugInfo.ValidTokenCount}");
        builder.AppendLine($"tokenIds[0:{Math.Min(previewTokenCount, debugInfo.InputIds.Length)}]={FormatSlice(debugInfo.InputIds, previewTokenCount)}");
        builder.AppendLine($"attentionMask[0:{Math.Min(previewTokenCount, debugInfo.AttentionMask.Length)}]={FormatSlice(debugInfo.AttentionMask, previewTokenCount)}");
        builder.AppendLine($"t5Hidden={DescribeStats(debugInfo.HiddenStates)}");
        builder.Append(FormatHiddenPreview(debugInfo.HiddenStates, debugInfo.ValidTokenCount, previewHiddenTokenCount, previewHiddenDimensionCount));
        return builder.ToString();
    }

    public static string DebugFirstDiTStep(
        string model,
        string prompt,
        float seconds = 7f,
        int steps = 8,
        float sigmaMax = 1f,
        int? seed = null,
        string? provider = null,
        int previewTokenCount = 32,
        int previewHiddenTokenCount = 4,
        int previewHiddenDimensionCount = 8)
    {
        if (seconds <= 0f)
            throw new ArgumentOutOfRangeException(nameof(seconds), "The audio duration must be greater than 0.");
        if (steps <= 0)
            throw new ArgumentOutOfRangeException(nameof(steps), "The number of sampling steps must be greater than 0.");
        if (sigmaMax is < 0.01f or > 1f)
            throw new ArgumentOutOfRangeException(nameof(sigmaMax), "The sigmaMax of rf_denoiser must be between 0.01 and 1.0.");
        if (previewTokenCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(previewTokenCount), "The token preview count must be greater than 0.");
        if (previewHiddenTokenCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(previewHiddenTokenCount), "The hidden-token preview count must be greater than 0.");
        if (previewHiddenDimensionCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(previewHiddenDimensionCount), "The hidden-dimension preview count must be greater than 0.");

        var runtime = new StableAudio(model, provider);
        WarnIfSmallSfxDuration(runtime.spec, seconds, 0f);

        int latentLength = Math.Max(1, (int)Math.Ceiling(seconds * SampleRate / SampleCompression));
        // Decoder SWA requires an even latentLength for both same-s and same-l decoders (see file header NOTE).
        if ((latentLength & 1) != 0)
            latentLength++;

        var promptDebug = runtime.EncodePromptDebug(runtime.bundle, prompt);
        var conditioned = promptDebug.Condition;
        var zeroCondition = CreateZeroCondition();
        var localAddCond = CreateLocalAddCondition(latentLength);
        var secondsTotal = CreateSecondsTotalCondition(seconds);
        var zeroSecondsTotal = CreateSecondsTotalCondition(0f);
        var timeSchedule = BuildPingPongSchedule(steps, sigmaMax, latentLength);
        float currentT = timeSchedule[0];
        float nextT = timeSchedule[1];
        float flippedT = 1f - currentT;
        float deltaT = currentT - nextT;
        var noiseRng = seed.HasValue ? new Random(seed.Value) : Random.Shared;
        var latent = CreateGaussianLatent(latentLength, noiseRng);
        var secondaryNoiseRng = seed.HasValue ? new Random(unchecked(seed.Value + 1)) : Random.Shared;
        var randomLatent2 = CreateGaussianLatent(latentLength, secondaryNoiseRng);
        var zeroLatent = new float[latent.Length];
        var negativeLatent = ScaleArray(latent, -1f);

        var velocity = RunDiT(
            runtime.bundle.DitSession,
            latent,
            latentLength,
            currentT,
            conditioned.HiddenStates,
            conditioned.Mask,
            secondsTotal,
            localAddCond);
        var flippedVelocity = RunDiT(
            runtime.bundle.DitSession,
            latent,
            latentLength,
            flippedT,
            conditioned.HiddenStates,
            conditioned.Mask,
            secondsTotal,
            localAddCond);
        var zeroSecondsVelocity = RunDiT(
            runtime.bundle.DitSession,
            latent,
            latentLength,
            currentT,
            conditioned.HiddenStates,
            conditioned.Mask,
            zeroSecondsTotal,
            localAddCond);
        var zeroConditionVelocity = RunDiT(
            runtime.bundle.DitSession,
            latent,
            latentLength,
            currentT,
            zeroCondition.HiddenStates,
            zeroCondition.Mask,
            secondsTotal,
            localAddCond);

        var denoised = ComputeDenoised(latent, velocity, currentT);
        var flippedDenoised = ComputeDenoised(latent, flippedVelocity, flippedT);
        var zeroSecondsDenoised = ComputeDenoised(latent, zeroSecondsVelocity, currentT);
        var zeroConditionDenoised = ComputeDenoised(latent, zeroConditionVelocity, currentT);
        float[] latentScaleScan = [0.25f, 0.5f, 1f, 2f];
        var zeroLatentVelocity = RunDiT(
            runtime.bundle.DitSession,
            zeroLatent,
            latentLength,
            currentT,
            conditioned.HiddenStates,
            conditioned.Mask,
            secondsTotal,
            localAddCond);
        var negativeLatentVelocity = RunDiT(
            runtime.bundle.DitSession,
            negativeLatent,
            latentLength,
            currentT,
            conditioned.HiddenStates,
            conditioned.Mask,
            secondsTotal,
            localAddCond);
        var randomLatent2Velocity = RunDiT(
            runtime.bundle.DitSession,
            randomLatent2,
            latentLength,
            currentT,
            conditioned.HiddenStates,
            conditioned.Mask,
            secondsTotal,
            localAddCond);
        var zeroLatentDenoised = ComputeDenoised(zeroLatent, zeroLatentVelocity, currentT);
        var negativeLatentDenoised = ComputeDenoised(negativeLatent, negativeLatentVelocity, currentT);
        var randomLatent2Denoised = ComputeDenoised(randomLatent2, randomLatent2Velocity, currentT);
        var denoisedMinusDeltaT = ComputeDenoisedWithDelta(latent, velocity, deltaT);
        var denoisedPlusDeltaT = ComputeDenoisedWithPositiveDelta(latent, velocity, deltaT);
        var denoisedPlusT = ComputeDenoisedWithPositiveTime(latent, velocity, currentT);

        var builder = new StringBuilder();
        builder.AppendLine($"model={runtime.model}");
        builder.AppendLine($"provider={DescribeProviderPlan(runtime.provider)}");
        builder.AppendLine($"seconds={seconds:0.###}");
        builder.AppendLine($"steps={steps}");
        builder.AppendLine($"sigmaMax={sigmaMax:0.###}");
        builder.AppendLine($"seed={(seed.HasValue ? seed.Value.ToString() : "random")}");
        builder.AppendLine($"latentLength={latentLength}");
        builder.AppendLine($"schedule={FormatFloatSlice(timeSchedule, timeSchedule.Length)}");
        builder.AppendLine($"step1={currentT:F6}->{nextT:F6}, deltaT={deltaT:F6}, flippedT={flippedT:F6}");
        builder.AppendLine($"normalizedPrompt={EscapeForLog(promptDebug.NormalizedPrompt)}");
        builder.AppendLine($"validTokenCount={promptDebug.ValidTokenCount}");
        builder.AppendLine($"tokenIds[0:{Math.Min(previewTokenCount, promptDebug.InputIds.Length)}]={FormatSlice(promptDebug.InputIds, previewTokenCount)}");
        builder.AppendLine($"attentionMask[0:{Math.Min(previewTokenCount, promptDebug.AttentionMask.Length)}]={FormatSlice(promptDebug.AttentionMask, previewTokenCount)}");
        builder.AppendLine($"t5Hidden={DescribeStats(promptDebug.HiddenStates)}");
        builder.AppendLine($"t5Mask={DescribeStats(conditioned.Mask.Buffer.Span)}");
        builder.AppendLine($"secondsTotal={DescribeStats(secondsTotal.Buffer.Span)}");
        builder.AppendLine($"secondsTotalZero={DescribeStats(zeroSecondsTotal.Buffer.Span)}");
        builder.AppendLine($"localAddCond={DescribeStats(localAddCond.Buffer.Span)}");
        builder.AppendLine($"latent={DescribeStats(latent)}");
        builder.AppendLine(FormatHiddenPreview(promptDebug.HiddenStates, promptDebug.ValidTokenCount, previewHiddenTokenCount, previewHiddenDimensionCount));
        builder.AppendLine();
        builder.AppendLine($"velocity@t={DescribeStats(velocity)}");
        builder.AppendLine($"velocity@flipT={DescribeStats(flippedVelocity)}");
        builder.AppendLine($"velocity@t_seconds0={DescribeStats(zeroSecondsVelocity)}");
        builder.AppendLine($"velocity@t_uncondZero={DescribeStats(zeroConditionVelocity)}");
        builder.AppendLine(DescribeArrayDifference("velocity@t vs flipT", velocity, flippedVelocity));
        builder.AppendLine(DescribeArrayDifference("velocity@t vs seconds0", velocity, zeroSecondsVelocity));
        builder.AppendLine(DescribeArrayDifference("velocity@t vs uncondZero", velocity, zeroConditionVelocity));
        builder.AppendLine($"denoised@t={DescribeStats(denoised)}");
        builder.AppendLine($"denoised@flipT={DescribeStats(flippedDenoised)}");
        builder.AppendLine($"denoised@t_seconds0={DescribeStats(zeroSecondsDenoised)}");
        builder.AppendLine($"denoised@t_uncondZero={DescribeStats(zeroConditionDenoised)}");
        builder.AppendLine(DescribeArrayDifference("denoised@t vs flipT", denoised, flippedDenoised));
        builder.AppendLine(DescribeArrayDifference("denoised@t vs seconds0", denoised, zeroSecondsDenoised));
        builder.AppendLine(DescribeArrayDifference("denoised@t vs uncondZero", denoised, zeroConditionDenoised));
        builder.AppendLine();
        builder.AppendLine("xVariantScan:");
        AppendInputVariantDebug(builder, "x=latent", latent, velocity, denoised, velocity, denoised);
        AppendInputVariantDebug(builder, "x=0", zeroLatent, zeroLatentVelocity, zeroLatentDenoised, velocity, denoised);
        AppendInputVariantDebug(builder, "x=-latent", negativeLatent, negativeLatentVelocity, negativeLatentDenoised, velocity, denoised);
        AppendInputVariantDebug(builder, "x=random2", randomLatent2, randomLatent2Velocity, randomLatent2Denoised, velocity, denoised);
        builder.AppendLine();
        builder.AppendLine("latentScaleScan:");

        foreach (float latentScale in latentScaleScan)
        {
            float[] scaledLatent = latentScale == 1f ? latent : ScaleArray(latent, latentScale);
            float[] scaledVelocity = latentScale == 1f
                ? velocity
                : RunDiT(
                    runtime.bundle.DitSession,
                    scaledLatent,
                    latentLength,
                    currentT,
                    conditioned.HiddenStates,
                    conditioned.Mask,
                    secondsTotal,
                    localAddCond);
            float[] scaledDenoised = latentScale == 1f
                ? denoised
                : ComputeDenoised(scaledLatent, scaledVelocity, currentT);

            builder.AppendLine($"latentScale={latentScale:F2}");
            builder.AppendLine($"  latent={DescribeStats(scaledLatent)}");
            builder.AppendLine($"  velocity={DescribeStats(scaledVelocity)}");
            builder.AppendLine($"  velocity vs scale1={DescribeArrayDifference("diff", velocity, scaledVelocity)}");
            builder.AppendLine($"  denoised={DescribeStats(scaledDenoised)}");
            builder.AppendLine($"  denoised vs scale1={DescribeArrayDifference("diff", denoised, scaledDenoised)}");
        }

        builder.AppendLine();
        builder.AppendLine("updateFormulaScan:");
        builder.AppendLine($"latent - t * velocity={DescribeStats(denoised)}");
        builder.AppendLine($"latent + t * velocity={DescribeStats(denoisedPlusT)}");
        builder.AppendLine($"latent - deltaT * velocity={DescribeStats(denoisedMinusDeltaT)}");
        builder.AppendLine($"latent + deltaT * velocity={DescribeStats(denoisedPlusDeltaT)}");
        builder.AppendLine(DescribeArrayDifference("minusT vs plusT", denoised, denoisedPlusT));
        builder.AppendLine(DescribeArrayDifference("minusT vs minusDeltaT", denoised, denoisedMinusDeltaT));
        builder.Append(DescribeArrayDifference("minusT vs plusDeltaT", denoised, denoisedPlusDeltaT));

        return builder.ToString();
    }

    public StableAudio(string model, string? provider = "cpu")
    {
        if (string.IsNullOrWhiteSpace(model))
            throw new ArgumentException("The model name cannot be empty.", nameof(model));

        this.model = model;
        this.provider = provider;
        spec = ResolveModelSpec(model, provider);
        bundle = CreateOrGetBundle(spec, provider);
    }

    StableAudio(ModelSpec spec, string? provider)
    {
        model = spec.CacheKey;
        this.provider = provider;
        this.spec = spec;
        bundle = CreateOrGetBundle(spec, provider);
    }

    static ModelBundle CreateOrGetBundle(ModelSpec spec, string? provider)
    {
        var lazyBundle = Bundles.GetOrAdd(
            spec.CacheKey,
            _ => new Lazy<ModelBundle>(
                () => LoadBundle((spec, provider)),
                LazyThreadSafetyMode.ExecutionAndPublication));
        try
        {
            return lazyBundle.Value;
        }
        catch
        {
            // A Lazy caches exceptions forever; evict the failed entry so a later attempt reloads
            // instead of rethrowing the stale failure.
            Bundles.TryRemove(spec.CacheKey, out _);
            throw;
        }
    }

    /// <summary>
    /// Disposes the shared session bundle for this instance and removes it from the cache.
    /// Any other StableAudio instance sharing the same bundle becomes unusable afterwards;
    /// the application is expected to unload through this path before dropping the context.
    /// </summary>
    public void Release()
    {
        if (Bundles.TryRemove(spec.CacheKey, out var lazy) && lazy.IsValueCreated)
            lazy.Value.Dispose();
    }

    /// <summary>
    /// Creates a StableAudio instance from explicit model file paths. Tokenizer
    /// files are provided as explicit paths: tokenizer.model,
    /// tokenizer_config.json, special_tokens_map.json.
    /// </summary>
    public static StableAudio FromFiles(string ditPath, string decoderPath, string textEncoderPath, string tokenizerModelPath, string tokenizerConfigPath, string specialTokensMapPath, string? provider = "cpu")
    {
        if (string.IsNullOrWhiteSpace(ditPath))
            throw new ArgumentException("The DiT model path cannot be empty.", nameof(ditPath));
        if (string.IsNullOrWhiteSpace(decoderPath))
            throw new ArgumentException("The decoder model path cannot be empty.", nameof(decoderPath));
        if (string.IsNullOrWhiteSpace(textEncoderPath))
            throw new ArgumentException("The text encoder model path cannot be empty.", nameof(textEncoderPath));
        if (string.IsNullOrWhiteSpace(tokenizerModelPath))
            throw new ArgumentException("The tokenizer.model path cannot be empty.", nameof(tokenizerModelPath));
        if (string.IsNullOrWhiteSpace(tokenizerConfigPath))
            throw new ArgumentException("The tokenizer_config.json path cannot be empty.", nameof(tokenizerConfigPath));
        if (string.IsNullOrWhiteSpace(specialTokensMapPath))
            throw new ArgumentException("The special_tokens_map.json path cannot be empty.", nameof(specialTokensMapPath));

        return new StableAudio(CreateSpecFromFiles(ditPath, decoderPath, textEncoderPath, tokenizerModelPath, tokenizerConfigPath, specialTokensMapPath, provider), provider);
    }

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
    {
        if (seconds <= 0f)
            throw new ArgumentOutOfRangeException(nameof(seconds), "The audio duration must be greater than 0.");
        if (steps <= 0)
            throw new ArgumentOutOfRangeException(nameof(steps), "The number of sampling steps must be greater than 0.");
        if (sigmaMax is < 0.01f or > 1f)
            throw new ArgumentOutOfRangeException(nameof(sigmaMax), "The sigmaMax of rf_denoiser should be between 0.01 and 1.0.");
        if (apgScale < 0f || apgScale > 1f)
            throw new ArgumentOutOfRangeException(nameof(apgScale), "The APG coefficient should be between 0 and 1.");
        if (decoderScale <= 0f)
            throw new ArgumentOutOfRangeException(nameof(decoderScale), "The latest scaling factor of the decoder must be greater than 0.");
        if (durationPaddingSeconds < 0f)
            throw new ArgumentOutOfRangeException(nameof(durationPaddingSeconds), "The duration padding cannot be negative.");

        WarnIfSmallSfxDuration(spec, seconds, durationPaddingSeconds);

        float generationSeconds = seconds + durationPaddingSeconds;
        int latentLength = Math.Max(1, (int)Math.Ceiling(generationSeconds * SampleRate / SampleCompression));
        // Decoder SWA requires an even latentLength for both same-s and same-l decoders (see file header NOTE).
        if ((latentLength & 1) != 0)
            latentLength++;

        WarnIfShortLatent(generationSeconds, latentLength, EnableDebugOutput);

        var initialNoiseRng = seed.HasValue ? new Random(seed.Value) : Random.Shared;
        var renoiseRng = seed.HasValue ? new Random(unchecked(seed.Value + 1)) : Random.Shared;

        var conditioned = EncodePrompt(bundle, prompt);
        bool useGuidance = cfgScale != 1f;
        var unconditioned = useGuidance
            ? (negativePrompt is null ? CreateZeroCondition() : EncodePrompt(bundle, negativePrompt))
            : conditioned;

        float conditionedSeconds = seconds;
        var latent = CreateGaussianLatent(latentLength, initialNoiseRng);
        var localAddCond = CreateLocalAddCondition(latentLength);
        var secondsTensor = CreateSecondsTotalCondition(conditionedSeconds);
        var timeSchedule = BuildPingPongSchedule(steps, sigmaMax, latentLength);
        var conditionedRunContext = new DiTRunContext(latent, latentLength, conditioned, secondsTensor, localAddCond);
        DiTRunContext? unconditionedRunContext = useGuidance
            ? new DiTRunContext(latent, latentLength, unconditioned, secondsTensor, localAddCond)
            : null;
        float[] denoisedBuffer = ArrayPool<float>.Shared.Rent(latent.Length);
        float[] stepNoiseBuffer = ArrayPool<float>.Shared.Rent(latent.Length);
        float[]? guidanceDiffBuffer = useGuidance ? ArrayPool<float>.Shared.Rent(latent.Length) : null;
        float[]? guidanceWorkBuffer = useGuidance ? ArrayPool<float>.Shared.Rent(latent.Length) : null;
        float[]? decoderBuffer = decoderScale == 1f && !EnableDebugOutput ? null : ArrayPool<float>.Shared.Rent(latent.Length);

        try
        {
            if (EnableDebugOutput)
            {
                Debug.WriteLine($"[StableAudio] model={model}, seconds={seconds}, conditionedSeconds={conditionedSeconds}, generationSeconds={generationSeconds}, steps={steps}, cfgScale={cfgScale}, sigmaMax={sigmaMax}, apgScale={apgScale}, decoderScale={decoderScale}, durationPaddingSeconds={durationPaddingSeconds}, truncateOutputToDuration={truncateOutputToDuration}, latentLength={latentLength}");
                Debug.WriteLine("[StableAudio] sampler=deltaT-pingpong");
                Debug.WriteLine($"[StableAudio] provider: {DescribeProviderPlan(provider)}");
                Debug.WriteLine($"[StableAudio] promptHidden: {DescribeStats(conditioned.HiddenStates.Buffer.Span)}");
                Debug.WriteLine($"[StableAudio] promptMask: {DescribeStats(conditioned.Mask.Buffer.Span)}");
                Debug.WriteLine($"[StableAudio] secondsTotal: {DescribeStats(secondsTensor.Buffer.Span)}");
                Debug.WriteLine($"[StableAudio] localAddCond: {DescribeStats(localAddCond.Buffer.Span)}");
                Debug.WriteLine($"[StableAudio] schedule: {string.Join(" -> ", Array.ConvertAll(timeSchedule, static value => value.ToString("F4")))}");
                Debug.WriteLine($"[StableAudio] initLatents: {DescribeStats(latent)}");
            }

            for (int stepIndex = 0; stepIndex < steps; stepIndex++)
            {
                float currentT = timeSchedule[stepIndex];
                float nextT = timeSchedule[stepIndex + 1];
                float deltaT = currentT - nextT;
                var conditionedVelocity = conditionedRunContext.Run(bundle.DitSession, currentT);

                float[]? unconditionedVelocity = null;
                if (useGuidance)
                {
                    unconditionedVelocity = unconditionedRunContext!.Run(bundle.DitSession, currentT);
                    ApplyOfficialGuidance(
                        latent,
                        conditionedVelocity,
                        unconditionedVelocity,
                        currentT,
                        cfgScale,
                        apgScale,
                        denoisedBuffer,
                        guidanceDiffBuffer!,
                        guidanceWorkBuffer!);
                }
                else
                {
                    ComputeDenoisedInto(latent, conditionedVelocity, currentT, denoisedBuffer);
                }

                if (EnableDebugOutput)
                {
                    float[]? conditionedDenoised = useGuidance
                        ? ComputeDenoised(latent, conditionedVelocity, currentT)
                        : null;
                    float[]? unconditionedDenoised = unconditionedVelocity is not null
                        ? ComputeDenoised(latent, unconditionedVelocity, currentT)
                        : null;
                    float flippedT = 1f - currentT;
                    float[]? flippedVelocity = !useGuidance
                        ? conditionedRunContext.Run(bundle.DitSession, flippedT)
                        : null;
                    float[]? flippedDenoised = flippedVelocity is not null
                        ? ComputeDenoised(latent, flippedVelocity, flippedT)
                        : null;
                    var latentMinusVelocity = ComputeLatentMinusVelocity(latent, conditionedVelocity);
                    var velocityOverTime = ComputeVelocityOverTime(conditionedVelocity, currentT);
                    var latentMinusDeltaTVelocity = ComputeDenoisedWithDelta(latent, conditionedVelocity, deltaT);
                    var latentMinusTimeVelocity = ComputeDenoised(latent, conditionedVelocity, currentT);
                    var latentPlusDeltaTVelocity = ComputeDenoisedWithPositiveDelta(latent, conditionedVelocity, deltaT);
                    var latentMinusOneMinusTimeVelocity = ComputeDenoisedWithOneMinusTime(latent, conditionedVelocity, currentT);
                    var latentPlusTimeVelocity = ComputeDenoisedWithPositiveTime(latent, conditionedVelocity, currentT);

                    Debug.WriteLine($"[StableAudio] step={stepIndex + 1}/{steps}, t={currentT:F4}->{nextT:F4}, deltaT={deltaT:F4}, velocity: {DescribeStats(conditionedVelocity)}");
                    if (flippedVelocity is not null)
                    {
                        Debug.WriteLine($"[StableAudio] step={stepIndex + 1}/{steps}, tFlip={flippedT:F4}, flippedVelocity: {DescribeStats(flippedVelocity)}");
                        Debug.WriteLine($"[StableAudio] step={stepIndex + 1}/{steps}, deltaT={deltaT:F4}, flippedDenoised: {DescribeStats(flippedDenoised!)}");
                    }
                    Debug.WriteLine($"[StableAudio] step={stepIndex + 1}/{steps}, t={currentT:F4}->{nextT:F4}, latentMinusVelocity: {DescribeStats(latentMinusVelocity)}");
                    Debug.WriteLine($"[StableAudio] step={stepIndex + 1}/{steps}, t={currentT:F4}->{nextT:F4}, velocityOverTime: {DescribeStats(velocityOverTime)}");
                    Debug.WriteLine($"[StableAudio] step={stepIndex + 1}/{steps}, t={currentT:F4}->{nextT:F4}, latentMinusTimeVelocity: {DescribeStats(latentMinusTimeVelocity)}");
                    Debug.WriteLine($"[StableAudio] step={stepIndex + 1}/{steps}, deltaT={deltaT:F4}, latentMinusDeltaTVelocity: {DescribeStats(latentMinusDeltaTVelocity)}");
                    Debug.WriteLine($"[StableAudio] step={stepIndex + 1}/{steps}, deltaT={deltaT:F4}, latentPlusDeltaTVelocity: {DescribeStats(latentPlusDeltaTVelocity)}");
                    Debug.WriteLine($"[StableAudio] step={stepIndex + 1}/{steps}, t={currentT:F4}->{nextT:F4}, latentMinusOneMinusTimeVelocity: {DescribeStats(latentMinusOneMinusTimeVelocity)}");
                    Debug.WriteLine($"[StableAudio] step={stepIndex + 1}/{steps}, t={currentT:F4}->{nextT:F4}, latentPlusTimeVelocity: {DescribeStats(latentPlusTimeVelocity)}");

                    if (unconditionedVelocity != null)
                    {
                        Debug.WriteLine($"[StableAudio] step={stepIndex + 1}/{steps}, t={currentT:F4}->{nextT:F4}, uncondVelocity: {DescribeStats(unconditionedVelocity)}");
                        Debug.WriteLine($"[StableAudio] step={stepIndex + 1}/{steps}, t={currentT:F4}->{nextT:F4}, condDenoised: {DescribeStats(conditionedDenoised!)}");
                        Debug.WriteLine($"[StableAudio] step={stepIndex + 1}/{steps}, t={currentT:F4}->{nextT:F4}, uncondDenoised: {DescribeStats(unconditionedDenoised!)}");
                    }

                    Debug.WriteLine($"[StableAudio] step={stepIndex + 1}/{steps}, t={currentT:F4}->{nextT:F4}, denoised: {DescribeStats(denoisedBuffer.AsSpan(0, latent.Length))}");
                }

                if (stepIndex < steps - 1 && nextT > 0f)
                {
                    FillGaussian(stepNoiseBuffer.AsSpan(0, latent.Length), renoiseRng);
                    for (int i = 0; i < latent.Length; i++)
                        latent[i] = ((1f - nextT) * denoisedBuffer[i]) + (nextT * stepNoiseBuffer[i]);
                }
                else
                {
                    Array.Copy(denoisedBuffer, latent, latent.Length);
                }

                if (EnableDebugOutput)
                    Debug.WriteLine($"[StableAudio] step={stepIndex + 1}/{steps}, latent: {DescribeStats(latent)}");
            }

            if (EnableDebugOutput)
                Debug.WriteLine($"[StableAudio] finalLatents: {DescribeStats(latent)}");

            float[] decoderLatent = latent;
            if (decoderScale != 1f)
            {
                decoderLatent = decoderBuffer!;
                for (int i = 0; i < latent.Length; i++)
                    decoderLatent[i] = latent[i] * decoderScale;
            }

            var decoded = DecodeAudio(bundle.DecoderSession, decoderLatent, latentLength);
            if (EnableDebugOutput)
            {
                Debug.WriteLine($"[StableAudio] decoderLatentScale={decoderScale:F5}, decoderLatent={DescribeStats(decoderLatent.AsSpan(0, latent.Length))}");
                Debug.WriteLine($"[StableAudio] decodedRaw: {DescribeDecodedAudio(decoded)}");
                float[] decoderLatentScales = [0.09375f, 0.125f, 0.15625f, 0.1875f, 0.25f, 0.5f, 1f, 2f];
                foreach (float latentScale in decoderLatentScales)
                {
                    float[] scaledLatent = latent;
                    if (latentScale != 1f)
                    {
                        scaledLatent = decoderBuffer!;
                        for (int i = 0; i < latent.Length; i++)
                            scaledLatent[i] = latent[i] * latentScale;
                    }

                    var scaledDecoded = DecodeAudio(bundle.DecoderSession, scaledLatent, latentLength);
                    Debug.WriteLine($"[StableAudio] decoderScale={latentScale:F5}, latent={DescribeStats(scaledLatent.AsSpan(0, latent.Length))}");
                    Debug.WriteLine($"[StableAudio] decoderScale={latentScale:F3}, decodedRaw={DescribeDecodedAudio(scaledDecoded)}");
                }
            }

            float outputSeconds = truncateOutputToDuration ? seconds : generationSeconds;
            return EncodeWav(decoded.Samples, decoded.ChannelCount, SampleRate, outputSeconds);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(denoisedBuffer);
            ArrayPool<float>.Shared.Return(stepNoiseBuffer);
            if (guidanceDiffBuffer is not null)
                ArrayPool<float>.Shared.Return(guidanceDiffBuffer);
            if (guidanceWorkBuffer is not null)
                ArrayPool<float>.Shared.Return(guidanceWorkBuffer);
            if (decoderBuffer is not null)
                ArrayPool<float>.Shared.Return(decoderBuffer);
        }
    }
}
