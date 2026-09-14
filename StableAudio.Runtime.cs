// Copyright (c) SeasonEngine and contributors.
// Licensed under the MIT License.
// https://github.com/SeasonRealms/SeasonAudio
// SeasonAudio for Stable Audio Models

namespace Season.Audio;

public partial class StableAudio
{
    static InferenceSession CreateSession(string modelPath, string? provider, bool allowCpuFallback = false)
    {
        try
        {
            return CreateSessionCore(modelPath, provider);
        }
        catch (OnnxRuntimeException ex) when (allowCpuFallback && !string.Equals(provider, "cpu", StringComparison.OrdinalIgnoreCase))
        {
            Trace.WriteLine(
                $"[StableAudio] provider session creation failed for `{modelPath}` with provider `{provider ?? "auto"}`. " +
                $"Fallback to CPUExecutionProvider. Reason: {ex.Message}");
            return CreateSessionCore(modelPath, "cpu");
        }
    }

    static InferenceSession CreateSessionCore(string modelPath, string? provider)
    {
        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
        };

        foreach (var methodName in GetProviderMethodNames(provider))
        {
            var method = typeof(SessionOptions)
                            .GetMethods(BindingFlags.Instance | BindingFlags.Public)
                            .Where(m => m.Name == methodName)
                            .Where(m => m.GetParameters().All(p => p.ParameterType.IsValueType))
                            .OrderBy(m => m.GetParameters().Length)
                            .FirstOrDefault();
            if (method == null)
                continue;

            try
            {
                var args = method.GetParameters()
                    .Select(CreateDefaultValue)
                    .ToArray();

                // CoreML flags: static input shapes + ML Program. Dynamic-shape partitions
                // (the decoder's latent-length input, or the DiT when it is not routed to CPU)
                // then stay on CPU instead of triggering expensive per-shape CoreML recompiles,
                // and ML Program compiles with a much smaller memory peak than the legacy
                // NeuralNetwork spec. The flag value is passed via reflection so the same source
                // compiles against both the uint and CoreMLFlags overloads (ORT 1.26 vs 1.29).
                if (args.Length == 1 && methodName.Contains("CoreML", StringComparison.Ordinal))
                {
                    // Convert.ChangeType only accepts strings for enum targets; numeric values throw
                    // "Invalid cast from 'System.UInt32' to 'Microsoft.ML.OnnxRuntime.CoreMLFlags'".
                    // Build the flag value per parameter type instead: Enum.ToObject for the
                    // CoreMLFlags overload (ORT 1.29), plain Convert.ChangeType for the uint one (1.26).
                    uint flags = (uint)CoreMLFlags.COREML_FLAG_ONLY_ALLOW_STATIC_INPUT_SHAPES |
                                 (uint)CoreMLFlags.COREML_FLAG_CREATE_MLPROGRAM;
                    var parameterType = method.GetParameters()[0].ParameterType;
                    args[0] = parameterType.IsEnum
                        ? Enum.ToObject(parameterType, flags)
                        : Convert.ChangeType(flags, parameterType);
                }

                method.Invoke(options, args);
                break;
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[StableAudio] failed to append provider via `{methodName}`: {ex.Message}");
            }
        }

        return new InferenceSession(modelPath, options);
    }

    static void ValidateDiTSessionContract(InferenceSession session, string modelPath)
    {
        ValidateMetadata(session.InputMetadata, "x", typeof(float), new int?[] { 1, LatentChannels, null }, modelPath);
        ValidateMetadata(session.InputMetadata, "t", typeof(float), new int?[] { 1 }, modelPath);
        ValidateMetadata(session.InputMetadata, "t5_hidden", typeof(float), new int?[] { 1, MaxTextTokens, 768 }, modelPath);
        ValidateMetadata(session.InputMetadata, "t5_mask", typeof(float), new int?[] { 1, MaxTextTokens }, modelPath);
        ValidateMetadata(session.InputMetadata, "seconds_total", typeof(float), new int?[] { 1 }, modelPath);
        ValidateMetadata(session.InputMetadata, "local_add_cond", typeof(float), new int?[] { 1, InpaintConditionChannels, null }, modelPath);
        ValidateMetadata(session.OutputMetadata, "velocity", typeof(float), new int?[] { null, LatentChannels, null }, modelPath);
    }

    static void ValidateMetadata(
        IReadOnlyDictionary<string, NodeMetadata> metadataMap,
        string name,
        Type expectedElementType,
        params int?[] expectedDimensions) =>
        ValidateMetadata(metadataMap, name, expectedElementType, expectedDimensions, string.Empty);

    static void ValidateMetadata(
        IReadOnlyDictionary<string, NodeMetadata> metadataMap,
        string name,
        Type expectedElementType,
        int?[] expectedDimensions,
        string modelPath)
    {
        if (!metadataMap.TryGetValue(name, out var metadata))
            throw new InvalidDataException($"The ONNX input/output contract is missing `{name}`: {modelPath}");

        if (metadata.ElementType != expectedElementType)
        {
            throw new InvalidDataException(
                $"ONNX tensor `{name}` has the wrong element type. Expected `{expectedElementType.Name}`, got `{metadata.ElementType?.Name ?? "unknown"}`: {modelPath}");
        }

        int[] actualDimensions = metadata.Dimensions;
        if (actualDimensions.Length != expectedDimensions.Length)
        {
            throw new InvalidDataException(
                $"ONNX tensor `{name}` has the wrong rank. Expected {expectedDimensions.Length}, got {actualDimensions.Length}: {modelPath}");
        }

        for (int i = 0; i < expectedDimensions.Length; i++)
        {
            int? expected = expectedDimensions[i];
            if (expected.HasValue && actualDimensions[i] != expected.Value)
            {
                throw new InvalidDataException(
                    $"ONNX tensor `{name}` has a mismatch at dimension {i}. Expected {expected.Value}, got {actualDimensions[i]}: {modelPath}");
            }
        }
    }

    static void ValidateRunDiTInputs(
        float[] latent,
        int latentLength,
        DenseTensor<float> hiddenStates,
        DenseTensor<float> t5Mask,
        DenseTensor<float> secondsTotal,
        DenseTensor<float> localAddCond)
    {
        if (latent.Length != LatentChannels * latentLength)
            throw new InvalidDataException($"DiT input `x` has the wrong flattened length. Expected {LatentChannels * latentLength}, got {latent.Length}.");

        ValidateTensorShape(hiddenStates, "t5_hidden", 1, MaxTextTokens, 768);
        ValidateTensorShape(t5Mask, "t5_mask", 1, MaxTextTokens);
        ValidateTensorShape(secondsTotal, "seconds_total", 1);
        ValidateTensorShape(localAddCond, "local_add_cond", 1, InpaintConditionChannels, latentLength);
    }

    static void ValidateTensorShape<T>(DenseTensor<T> tensor, string name, params int[] expectedDimensions)
        where T : struct
    {
        var actualDimensions = tensor.Dimensions.ToArray();
        if (actualDimensions.Length != expectedDimensions.Length)
        {
            throw new InvalidDataException(
                $"DiT input `{name}` has the wrong rank. Expected {expectedDimensions.Length}, got {actualDimensions.Length}.");
        }

        for (int i = 0; i < expectedDimensions.Length; i++)
        {
            if (actualDimensions[i] != expectedDimensions[i])
            {
                throw new InvalidDataException(
                    $"DiT input `{name}` has a mismatch at dimension {i}. Expected {expectedDimensions[i]}, got {actualDimensions[i]}.");
            }
        }
    }

    static object? CreateDefaultValue(ParameterInfo parameter)
    {
        if (parameter.HasDefaultValue)
            return parameter.DefaultValue;

        return parameter.ParameterType.IsValueType
            ? Activator.CreateInstance(parameter.ParameterType)
            : null;
    }

    static string[] GetProviderMethodNames(string? provider)
    {
        return (provider ?? "auto").ToLowerInvariant() switch
        {
            "cpu" => Array.Empty<string>(),
            "dml" => new[] { "AppendExecutionProvider_DML" },
            "cuda" => new[] { "AppendExecutionProvider_CUDA" },
            "coreml" => new[] { "AppendExecutionProvider_CoreML" },
            "nnapi" => new[] { "AppendExecutionProvider_Nnapi", "AppendExecutionProvider_NNAPI" },
            _ when OperatingSystem.IsWindows() => new[] { "AppendExecutionProvider_DML", "AppendExecutionProvider_CUDA" },
            _ when OperatingSystem.IsAndroid() => new[] { "AppendExecutionProvider_Nnapi", "AppendExecutionProvider_NNAPI" },
            _ when OperatingSystem.IsIOS() || OperatingSystem.IsMacOS() => new[] { "AppendExecutionProvider_CoreML" },
            _ => new[] { "AppendExecutionProvider_CUDA" }
        };
    }

    static string DescribeProviderPlan(string? provider)
    {
        string requested = provider ?? "auto";
        string[] methods = GetProviderMethodNames(provider);

        if (methods.Length == 0)
            return $"requested={requested}, session=CPUExecutionProvider";

        return $"requested={requested}, session=ORT default + [{string.Join(" -> ", methods)}] fallback chain";
    }
}
