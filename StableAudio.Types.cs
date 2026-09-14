// Copyright (c) SeasonEngine and contributors.
// Licensed under the MIT License.
// https://github.com/SeasonRealms/SeasonAudio
// SeasonAudio for Stable Audio Models

namespace Season.Audio;

public partial class StableAudio
{
    sealed record ModelSpec(
        string DitPath,
        string DecoderPath,
        string TextEncoderPath,
        string TokenizerModelPath,
        string TokenizerConfigPath,
        string SpecialTokensMapPath,
        string? TokenizerJsonPath,
        string CacheKey);

    sealed record TextCondition(
        DenseTensor<float> HiddenStates,
        DenseTensor<float> Mask);

    sealed record PromptEncodingDebug(
        string NormalizedPrompt,
        long[] InputIds,
        long[] AttentionMask,
        float[] HiddenStates,
        int ValidTokenCount,
        TextCondition Condition);

    sealed record DecodedAudio(
        float[] Samples,
        int ChannelCount);

    sealed class ModelBundle : IDisposable
    {
        public ModelBundle(
            PromptTokenizer tokenizer,
            InferenceSession textEncoderSession,
            InferenceSession ditSession,
            InferenceSession decoderSession)
        {
            Tokenizer = tokenizer;
            TextEncoderSession = textEncoderSession;
            DitSession = ditSession;
            DecoderSession = decoderSession;
        }

        public PromptTokenizer Tokenizer { get; }

        public InferenceSession TextEncoderSession { get; }

        public InferenceSession DitSession { get; }

        public InferenceSession DecoderSession { get; }

        public void Dispose()
        {
            TextEncoderSession.Dispose();
            DitSession.Dispose();
            DecoderSession.Dispose();
        }
    }

    sealed class DiTRunContext
    {
        readonly float[] timestepData = new float[1];
        readonly DenseTensor<float> latentTensor;
        readonly DenseTensor<float> timestepTensor;
        readonly NamedOnnxValue[] inputs;

        public DiTRunContext(
            float[] latent,
            int latentLength,
            TextCondition condition,
            DenseTensor<float> secondsTotal,
            DenseTensor<float> localAddCond)
        {
            latentTensor = new DenseTensor<float>(latent, new[] { 1, LatentChannels, latentLength });
            timestepTensor = new DenseTensor<float>(timestepData, SingleValueDimensions);
            inputs =
            [
                NamedOnnxValue.CreateFromTensor("x", latentTensor),
                NamedOnnxValue.CreateFromTensor("t", timestepTensor),
                NamedOnnxValue.CreateFromTensor("t5_hidden", condition.HiddenStates),
                NamedOnnxValue.CreateFromTensor("t5_mask", condition.Mask),
                NamedOnnxValue.CreateFromTensor("seconds_total", secondsTotal),
                NamedOnnxValue.CreateFromTensor("local_add_cond", localAddCond)
            ];
        }

        public float[] Run(InferenceSession session, float timestep)
        {
            timestepData[0] = timestep;
            using var results = session.Run(inputs);
            return CopyRequiredFloatTensor(results, "velocity");
        }
    }

    sealed record TokenizerMetadata(
        Dictionary<string, int> SpecialTokens,
        int PadTokenId,
        string UnknownToken);

    sealed class PromptTokenizer
    {
        readonly Func<string, string> normalize;
        readonly Func<string, IReadOnlyList<int>> encodeToIds;

        public PromptTokenizer(
            int padTokenId,
            Func<string, string> normalize,
            Func<string, IReadOnlyList<int>> encodeToIds)
        {
            PadTokenId = padTokenId;
            this.normalize = normalize;
            this.encodeToIds = encodeToIds;
        }

        public int PadTokenId { get; }

        public string Normalize(string text) => normalize(text);

        public IReadOnlyList<int> EncodeToIds(string text) => encodeToIds(text);
    }

    readonly record struct BpeMergePair(string Left, string Right);

    sealed class FastBpeTokenizer
    {
        readonly Dictionary<string, int> vocabulary;
        readonly Dictionary<BpeMergePair, int> mergeRanks;
        readonly KeyValuePair<string, int>[] specialTokens;
        readonly Dictionary<byte, int> byteFallbackTokenIds;
        readonly Func<string, string> normalizer;
        readonly string unknownToken;
        readonly int unknownTokenId;
        readonly bool fuseUnknownTokens;
        readonly bool byteFallbackEnabled;

        public FastBpeTokenizer(
            Dictionary<string, int> vocabulary,
            Dictionary<BpeMergePair, int> mergeRanks,
            Dictionary<string, int> specialTokens,
            string unknownToken,
            int unknownTokenId,
            bool fuseUnknownTokens,
            bool byteFallbackEnabled,
            Func<string, string> normalizer)
        {
            this.vocabulary = vocabulary;
            this.mergeRanks = mergeRanks;
            this.normalizer = normalizer;
            this.unknownToken = unknownToken;
            this.unknownTokenId = unknownTokenId;
            this.fuseUnknownTokens = fuseUnknownTokens;
            this.byteFallbackEnabled = byteFallbackEnabled;
            this.specialTokens = specialTokens
                .OrderByDescending(static pair => pair.Key.Length)
                .ToArray();
            byteFallbackTokenIds = BuildByteFallbackTokenIds(vocabulary);
        }

        public IReadOnlyList<int> EncodeToIds(string text)
        {
            if (string.IsNullOrEmpty(text))
                return Array.Empty<int>();

            string normalized = normalizer(text);
            var ids = new List<int>(normalized.Length);
            foreach (var segment in SplitBySpecialTokens(normalized))
            {
                if (segment.IsSpecial)
                {
                    ids.Add(segment.TokenId);
                    continue;
                }

                EncodeSegment(segment.Text, ids);
            }

            return ids;
        }

        void EncodeSegment(string text, List<int> destination)
        {
            if (string.IsNullOrEmpty(text))
                return;

            var symbols = text.EnumerateRunes()
                .Select(static rune => rune.ToString())
                .ToList();

            ApplyBpeMerges(symbols);

            bool previousWasUnknown = false;
            foreach (string symbol in symbols)
            {
                if (vocabulary.TryGetValue(symbol, out int tokenId))
                {
                    destination.Add(tokenId);
                    previousWasUnknown = false;
                    continue;
                }

                if (byteFallbackEnabled && TryEncodeAsByteFallback(symbol, destination))
                {
                    previousWasUnknown = false;
                    continue;
                }

                if (!fuseUnknownTokens || !previousWasUnknown)
                    destination.Add(unknownTokenId);

                previousWasUnknown = true;
            }
        }

        void ApplyBpeMerges(List<string> symbols)
        {
            while (symbols.Count > 1)
            {
                int bestIndex = -1;
                int bestRank = int.MaxValue;

                for (int i = 0; i < symbols.Count - 1; i++)
                {
                    if (!mergeRanks.TryGetValue(new BpeMergePair(symbols[i], symbols[i + 1]), out int rank))
                        continue;

                    if (rank < bestRank)
                    {
                        bestRank = rank;
                        bestIndex = i;
                    }
                }

                if (bestIndex < 0)
                    break;

                symbols[bestIndex] += symbols[bestIndex + 1];
                symbols.RemoveAt(bestIndex + 1);
            }
        }

        bool TryEncodeAsByteFallback(string symbol, List<int> destination)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(symbol);
            var fallbackIds = new int[bytes.Length];

            for (int i = 0; i < bytes.Length; i++)
            {
                if (!byteFallbackTokenIds.TryGetValue(bytes[i], out int tokenId))
                    return false;

                fallbackIds[i] = tokenId;
            }

            destination.AddRange(fallbackIds);
            return true;
        }

        IEnumerable<TokenSegment> SplitBySpecialTokens(string text)
        {
            if (specialTokens.Length == 0)
            {
                yield return new TokenSegment(text, false, 0);
                yield break;
            }

            var buffer = new StringBuilder();

            for (int index = 0; index < text.Length;)
            {
                bool matched = false;
                foreach (var specialToken in specialTokens)
                {
                    if (!text.AsSpan(index).StartsWith(specialToken.Key.AsSpan(), StringComparison.Ordinal))
                        continue;

                    if (buffer.Length > 0)
                    {
                        yield return new TokenSegment(buffer.ToString(), false, 0);
                        buffer.Clear();
                    }

                    yield return new TokenSegment(specialToken.Key, true, specialToken.Value);
                    index += specialToken.Key.Length;
                    matched = true;
                    break;
                }

                if (matched)
                    continue;

                var rune = Rune.GetRuneAt(text, index);
                buffer.Append(rune.ToString());
                index += rune.Utf16SequenceLength;
            }

            if (buffer.Length > 0)
                yield return new TokenSegment(buffer.ToString(), false, 0);
        }

        static Dictionary<byte, int> BuildByteFallbackTokenIds(Dictionary<string, int> vocabulary)
        {
            var result = new Dictionary<byte, int>();
            foreach (var pair in vocabulary)
            {
                if (!TryParseByteFallbackToken(pair.Key, out byte value))
                    continue;

                result[value] = pair.Value;
            }

            return result;
        }

        static bool TryParseByteFallbackToken(string token, out byte value)
        {
            value = 0;
            if (token.Length != 6 || !token.StartsWith("<0x", StringComparison.Ordinal) || token[5] != '>')
                return false;

            return byte.TryParse(token.AsSpan(3, 2), NumberStyles.HexNumber, CultureInfo.InvariantCulture, out value);
        }

        readonly record struct TokenSegment(string Text, bool IsSpecial, int TokenId);
    }
}
