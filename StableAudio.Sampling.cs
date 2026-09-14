// Copyright (c) SeasonEngine and contributors.
// Licensed under the MIT License.
// https://github.com/SeasonRealms/SeasonAudio
// SeasonAudio for Stable Audio Models

namespace Season.Audio;

public partial class StableAudio
{
    static float[] CreateGaussianLatent(int latentLength, Random random)
    {
        int total = LatentChannels * latentLength;
        var data = new float[total];
        FillGaussian(data, random);
        return data;
    }

    static void FillGaussian(Span<float> destination, Random random)
    {
        int index = 0;
        while (index < destination.Length)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();
            double radius = Math.Sqrt(-2.0 * Math.Log(u1));
            double theta = 2.0 * Math.PI * u2;

            destination[index++] = (float)(radius * Math.Cos(theta));
            if (index < destination.Length)
                destination[index++] = (float)(radius * Math.Sin(theta));
        }
    }

    static DenseTensor<float> CreateLocalAddCondition(int latentLength)
    {
        var data = new float[InpaintConditionChannels * latentLength];
        return new DenseTensor<float>(data, new[] { 1, InpaintConditionChannels, latentLength });
    }

    static DenseTensor<float> CreateSecondsTotalCondition(float seconds)
    {
        float clampedSeconds = Math.Clamp(seconds, 0f, MaxConditionSeconds);
        float[] data = [clampedSeconds];
        return new DenseTensor<float>(data, SingleValueDimensions);
    }

    static void ApplyOfficialGuidance(
        float[] latent,
        float[] conditionedVelocity,
        float[] unconditionedVelocity,
        float sigma,
        float cfgScale,
        float apgScale,
        float[] destination,
        float[] diffBuffer,
        float[] guidanceBuffer)
    {
        ComputeDenoisedInto(latent, conditionedVelocity, sigma, destination);
        for (int i = 0; i < latent.Length; i++)
            diffBuffer[i] = sigma * (unconditionedVelocity[i] - conditionedVelocity[i]);

        float[] sourceBuffer = diffBuffer;
        if (apgScale <= 0f)
        {
            Array.Copy(diffBuffer, guidanceBuffer, latent.Length);
            sourceBuffer = guidanceBuffer;
        }
        else
        {
            double normSquared = 0d;
            for (int i = 0; i < latent.Length; i++)
                normSquared += destination[i] * destination[i];

            float inverseNorm = normSquared <= 1e-8d
                ? 0f
                : (float)(1.0 / Math.Sqrt(normSquared));

            if (apgScale >= 1f)
            {
                ProjectOrthogonal(diffBuffer, destination, inverseNorm, guidanceBuffer);
            }
            else
            {
                ProjectOrthogonal(diffBuffer, destination, inverseNorm, guidanceBuffer);
                for (int i = 0; i < latent.Length; i++)
                    guidanceBuffer[i] = (apgScale * guidanceBuffer[i]) + ((1f - apgScale) * diffBuffer[i]);
            }

            sourceBuffer = guidanceBuffer;
        }

        float guidanceAmount = cfgScale - 1f;
        for (int i = 0; i < latent.Length; i++)
            destination[i] += guidanceAmount * sourceBuffer[i];
    }

    static void ProjectOrthogonal(float[] diff, float[] conditionedDenoised, float inverseNorm, float[] destination)
    {
        if (inverseNorm == 0f)
        {
            Array.Copy(diff, destination, diff.Length);
            return;
        }

        double parallelScale = 0d;
        for (int i = 0; i < diff.Length; i++)
            parallelScale += diff[i] * (conditionedDenoised[i] * inverseNorm);

        for (int i = 0; i < diff.Length; i++)
        {
            float unit = conditionedDenoised[i] * inverseNorm;
            destination[i] = diff[i] - ((float)parallelScale * unit);
        }
    }

    static float[] ComputeDenoised(float[] latent, float[] velocity, float timestep)
    {
        var denoised = new float[latent.Length];
        ComputeDenoisedInto(latent, velocity, timestep, denoised);

        return denoised;
    }

    static void ComputeDenoisedInto(float[] latent, float[] velocity, float timestep, float[] destination)
    {
        for (int i = 0; i < latent.Length; i++)
            destination[i] = latent[i] - (timestep * velocity[i]);
    }

    static float[] ComputeLatentMinusVelocity(float[] latent, float[] velocity)
    {
        var candidate = new float[latent.Length];
        for (int i = 0; i < latent.Length; i++)
            candidate[i] = latent[i] - velocity[i];

        return candidate;
    }

    static float[] ComputeVelocityOverTime(float[] velocity, float timestep)
    {
        float divisor = MathF.Max(MathF.Abs(timestep), 1e-4f);
        var candidate = new float[velocity.Length];
        for (int i = 0; i < velocity.Length; i++)
            candidate[i] = velocity[i] / divisor;

        return candidate;
    }

    static float[] ComputeDenoisedWithOneMinusTime(float[] latent, float[] velocity, float timestep)
    {
        float factor = 1f - timestep;
        var candidate = new float[latent.Length];
        for (int i = 0; i < latent.Length; i++)
            candidate[i] = latent[i] - (factor * velocity[i]);

        return candidate;
    }

    static float[] ComputeDenoisedWithPositiveTime(float[] latent, float[] velocity, float timestep)
    {
        var candidate = new float[latent.Length];
        for (int i = 0; i < latent.Length; i++)
            candidate[i] = latent[i] + (timestep * velocity[i]);

        return candidate;
    }

    static float[] BuildPingPongSchedule(int steps, float sigmaMax, int latentLength)
    {
        var schedule = new float[steps + 1];
        for (int i = 0; i <= steps; i++)
        {
            float linear = sigmaMax * (1f - (i / (float)steps));
            schedule[i] = ApplyDistributionShift(linear, latentLength);
        }

        schedule[0] = sigmaMax;
        schedule[^1] = 0f;
        return schedule;
    }

    static float ApplyDistributionShift(float t, int latentLength)
    {
        if (t <= 0f)
            return 0f;
        if (t >= 1f)
            return 1f;

        int clampedLength = Math.Clamp(latentLength, DitTrainedMinLatentLength, DitMaxLatentLength);
        float mu = -(DistributionShiftBase +
            ((DistributionShiftMax - DistributionShiftBase) *
            (clampedLength - DitTrainedMinLatentLength) /
            (float)(DitMaxLatentLength - DitTrainedMinLatentLength)));

        double expMu = Math.Exp(mu);
        double odds = (1.0 / (1.0 - t)) - 1.0;
        return (float)(1.0 - (expMu / (expMu + odds)));
    }

    static void WarnIfShortLatent(float seconds, int latentLength, bool debug)
    {
        if (latentLength >= DitTrainedMinLatentLength)
            return;

        string message =
            $"[StableAudio] warning: latentLength={latentLength} (= {seconds:0.##}s) is below the DiT's trained minimum " +
            $"({DitTrainedMinLatentLength} ~= {((float)DitTrainedMinLatentLength * SampleCompression / SampleRate):0.0}s). " +
            "Engine runs; output quality is undefined.";

        Trace.WriteLine(message);
        if (debug)
            Debug.WriteLine(message);
    }

    static void WarnIfSmallSfxDuration(ModelSpec spec, float seconds, float durationPaddingSeconds)
    {
        if (!IsSmallSfxSpec(spec))
            return;

        float conditionedSeconds = seconds + durationPaddingSeconds;
        if (seconds <= 10f && conditionedSeconds <= 12f)
            return;

        string message =
            $"[StableAudio] warning: `small-sfx` is better suited for short sound effects / one-shots. Current seconds={seconds:0.##}, " +
            $"conditionedSeconds={conditionedSeconds:0.##}; it is more likely to degrade into repeated texture or noise. Prefer trying 3-7 seconds first, " +
            "and keep durationPaddingSeconds within 0-1 seconds.";
        Trace.WriteLine(message);
        Debug.WriteLine(message);
    }

    static bool IsSmallSfxSpec(ModelSpec spec)
    {
        string key = spec.CacheKey.Replace('\\', '/').ToLowerInvariant();
        return key.Contains("sa3-sm-sfx", StringComparison.Ordinal) ||
               key.Contains("stable-audio-3-small-sfx", StringComparison.Ordinal);
    }

    static void Sanitize(float[] values)
    {
        for (int i = 0; i < values.Length; i++)
        {
            if (float.IsNaN(values[i]) || float.IsInfinity(values[i]))
                values[i] = 0f;
        }
    }

    static byte[] EncodeWav(float[] interleaved, int channelCount, int sampleRate, float seconds)
    {
        if (channelCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(channelCount));

        int maxFrameCount = Math.Max(1, (int)Math.Round(seconds * sampleRate));
        int frameCount = Math.Min(interleaved.Length / channelCount, maxFrameCount);
        int sampleCount = frameCount * channelCount;

        float peak = 0f;
        for (int i = 0; i < sampleCount; i++)
        {
            float value = interleaved[i];
            if (float.IsNaN(value) || float.IsInfinity(value))
                value = 0f;

            float abs = Math.Abs(value);
            if (abs > peak)
                peak = abs;

            interleaved[i] = value;
        }

        int dataSize = sampleCount * sizeof(short);
        using var stream = new MemoryStream(44 + dataSize);
        using var writer = new BinaryWriter(stream);

        writer.Write("RIFF"u8.ToArray());
        writer.Write(36 + dataSize);
        writer.Write("WAVE"u8.ToArray());
        writer.Write("fmt "u8.ToArray());
        writer.Write(16);
        writer.Write((short)1);
        writer.Write((short)channelCount);
        writer.Write(sampleRate);
        writer.Write(sampleRate * channelCount * sizeof(short));
        writer.Write((short)(channelCount * sizeof(short)));
        writer.Write((short)16);
        writer.Write("data"u8.ToArray());
        writer.Write(dataSize);

        bool looksLikePcm = peak > 2f;
        for (int i = 0; i < sampleCount; i++)
        {
            short sample = looksLikePcm
                ? (short)Math.Clamp(Math.Round(interleaved[i]), short.MinValue, short.MaxValue)
                : (short)Math.Round(Math.Clamp(interleaved[i], -1f, 1f) * short.MaxValue);
            writer.Write(sample);
        }

        writer.Flush();
        return stream.ToArray();
    }
}
