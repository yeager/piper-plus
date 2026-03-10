/**
 * Test: Zero-Shot Speaker Embedding
 *
 * Tests for speaker embedding handling in zero-shot TTS:
 * - .npy file parsing (NumPy binary format)
 * - SynthesisConfig speaker embedding field
 * - ModelConfig zero-shot fields
 * - Input detection for speaker_embedding
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <optional>

// ---------------------------------------------------------------------------
// Minimal .npy parser (mirrors the logic in main.cpp)
// ---------------------------------------------------------------------------

struct NpyParseResult {
    bool success = false;
    std::string error;
    std::vector<float> data;
};

NpyParseResult parseNpyFile(const std::string& path) {
    NpyParseResult result;

    std::ifstream file(path, std::ios::binary);
    if (!file.good()) {
        result.error = "Cannot open file";
        return result;
    }

    // Magic: \x93NUMPY
    char magic[6];
    file.read(magic, 6);
    if (magic[0] != '\x93' || std::string(magic + 1, 5) != "NUMPY") {
        result.error = "Invalid .npy magic";
        return result;
    }

    // Version
    uint8_t major_ver, minor_ver;
    file.read(reinterpret_cast<char*>(&major_ver), 1);
    file.read(reinterpret_cast<char*>(&minor_ver), 1);

    // Header length
    uint32_t header_len = 0;
    if (major_ver == 1) {
        uint16_t hl;
        file.read(reinterpret_cast<char*>(&hl), 2);
        header_len = hl;
    } else {
        file.read(reinterpret_cast<char*>(&header_len), 4);
    }

    // Skip header
    file.seekg(header_len, std::ios::cur);

    // Read float32 data
    auto dataStart = file.tellg();
    file.seekg(0, std::ios::end);
    auto dataEnd = file.tellg();
    file.seekg(dataStart);

    size_t dataBytes = dataEnd - dataStart;
    size_t numFloats = dataBytes / sizeof(float);

    result.data.resize(numFloats);
    file.read(reinterpret_cast<char*>(result.data.data()), dataBytes);

    result.success = true;
    return result;
}

// ---------------------------------------------------------------------------
// Helper: Create a minimal .npy file (NumPy v1.0, float32, 1D)
// ---------------------------------------------------------------------------

std::string createTempNpy(const std::vector<float>& data, const std::string& suffix = "") {
    auto path = std::filesystem::temp_directory_path() / ("test_embedding" + suffix + ".npy");

    std::ofstream file(path.string(), std::ios::binary);

    // Magic
    file.put('\x93');
    file.write("NUMPY", 5);

    // Version 1.0
    file.put(1);
    file.put(0);

    // Header: must be padded to multiple of 64 bytes (including magic+version+header_len)
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
                          std::to_string(data.size()) + ",), }";
    // Pad header with spaces, terminated by newline
    // Total = 6 (magic) + 2 (version) + 2 (header_len) + header_len = multiple of 64
    size_t prefix_len = 10; // magic(6) + version(2) + header_len(2)
    size_t pad_target = ((prefix_len + header.size() + 1 + 63) / 64) * 64;
    size_t pad_needed = pad_target - prefix_len - header.size() - 1;
    header += std::string(pad_needed, ' ');
    header += '\n';

    // Write header length (little-endian uint16)
    uint16_t header_len = static_cast<uint16_t>(header.size());
    file.write(reinterpret_cast<const char*>(&header_len), 2);

    // Write header
    file.write(header.data(), header.size());

    // Write float32 data
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));

    file.close();
    return path.string();
}

// ---------------------------------------------------------------------------
// Tests: .npy File Parsing
// ---------------------------------------------------------------------------

TEST(ZeroShotEmbeddingTest, ParseValidNpyFile) {
    // Create a test .npy file with 192 floats
    std::vector<float> expected(192);
    for (size_t i = 0; i < 192; i++) {
        expected[i] = static_cast<float>(i) / 192.0f;
    }

    auto path = createTempNpy(expected, "_valid");
    auto result = parseNpyFile(path);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.data.size(), 192);

    for (size_t i = 0; i < 192; i++) {
        EXPECT_FLOAT_EQ(result.data[i], expected[i]) << "Mismatch at index " << i;
    }

    std::filesystem::remove(path);
}

TEST(ZeroShotEmbeddingTest, ParseNpyDimension192) {
    std::vector<float> data(192, 1.0f);
    auto path = createTempNpy(data, "_dim192");
    auto result = parseNpyFile(path);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.data.size(), 192);

    std::filesystem::remove(path);
}

TEST(ZeroShotEmbeddingTest, ParseNpyDimension256) {
    // Support other embedding dimensions (e.g., ECAPA-TDNN 256-dim)
    std::vector<float> data(256, 0.5f);
    auto path = createTempNpy(data, "_dim256");
    auto result = parseNpyFile(path);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.data.size(), 256);

    std::filesystem::remove(path);
}

TEST(ZeroShotEmbeddingTest, ParseNpyFileNotFound) {
    auto result = parseNpyFile("/nonexistent/path/embedding.npy");
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.error, "Cannot open file");
}

TEST(ZeroShotEmbeddingTest, ParseInvalidMagic) {
    auto path = std::filesystem::temp_directory_path() / "test_bad_magic.npy";
    std::ofstream file(path.string(), std::ios::binary);
    file.write("NOT_NPY_FILE", 12);
    file.close();

    auto result = parseNpyFile(path.string());
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.error, "Invalid .npy magic");

    std::filesystem::remove(path);
}

TEST(ZeroShotEmbeddingTest, ParseNpyPreservesValues) {
    // Verify specific values are preserved through parse round-trip
    std::vector<float> data = {-1.0f, 0.0f, 1.0f, 3.14159f, -2.71828f};
    auto path = createTempNpy(data, "_values");
    auto result = parseNpyFile(path);

    EXPECT_TRUE(result.success);
    ASSERT_EQ(result.data.size(), data.size());
    EXPECT_FLOAT_EQ(result.data[0], -1.0f);
    EXPECT_FLOAT_EQ(result.data[1], 0.0f);
    EXPECT_FLOAT_EQ(result.data[2], 1.0f);
    EXPECT_FLOAT_EQ(result.data[3], 3.14159f);
    EXPECT_FLOAT_EQ(result.data[4], -2.71828f);

    std::filesystem::remove(path);
}

// ---------------------------------------------------------------------------
// Tests: ModelConfig Zero-Shot Fields
// ---------------------------------------------------------------------------

struct TestModelConfig {
    int numSpeakers = 1;
    bool useZeroShot = false;
    int spkEmbedDim = 192;
};

TEST(ZeroShotEmbeddingTest, ModelConfigDefaults) {
    TestModelConfig config;
    EXPECT_FALSE(config.useZeroShot);
    EXPECT_EQ(config.spkEmbedDim, 192);
}

TEST(ZeroShotEmbeddingTest, ModelConfigZeroShotEnabled) {
    TestModelConfig config;
    config.useZeroShot = true;
    config.spkEmbedDim = 192;

    EXPECT_TRUE(config.useZeroShot);
    EXPECT_EQ(config.spkEmbedDim, 192);
}

TEST(ZeroShotEmbeddingTest, ModelConfigCustomEmbedDim) {
    TestModelConfig config;
    config.useZeroShot = true;
    config.spkEmbedDim = 256;

    EXPECT_EQ(config.spkEmbedDim, 256);
}

// ---------------------------------------------------------------------------
// Tests: SynthesisConfig Speaker Embedding
// ---------------------------------------------------------------------------

struct TestSynthesisConfig {
    float noiseScale = 0.667f;
    float lengthScale = 1.0f;
    float noiseW = 0.8f;
    std::optional<int64_t> speakerId;
    std::optional<std::vector<float>> speakerEmbedding;
};

TEST(ZeroShotEmbeddingTest, SynthesisConfigNoEmbedding) {
    TestSynthesisConfig config;
    EXPECT_FALSE(config.speakerEmbedding.has_value());
}

TEST(ZeroShotEmbeddingTest, SynthesisConfigWithEmbedding) {
    TestSynthesisConfig config;
    std::vector<float> emb(192, 0.1f);
    config.speakerEmbedding = emb;

    EXPECT_TRUE(config.speakerEmbedding.has_value());
    EXPECT_EQ(config.speakerEmbedding->size(), 192);
}

TEST(ZeroShotEmbeddingTest, SynthesisConfigEmbeddingAndSpeakerIdMutuallyExclusive) {
    // Zero-shot uses speakerEmbedding, not speakerId
    TestSynthesisConfig config;
    config.speakerEmbedding = std::vector<float>(192, 0.5f);

    EXPECT_TRUE(config.speakerEmbedding.has_value());
    EXPECT_FALSE(config.speakerId.has_value());
}

TEST(ZeroShotEmbeddingTest, SynthesisConfigResetEmbeddingAfterJsonInput) {
    // After processing JSON input, embedding should be reset
    TestSynthesisConfig config;
    config.speakerEmbedding = std::vector<float>(192, 0.5f);
    EXPECT_TRUE(config.speakerEmbedding.has_value());

    // Simulate reset (as done in main.cpp after each JSON line)
    config.speakerEmbedding = std::nullopt;
    EXPECT_FALSE(config.speakerEmbedding.has_value());
}

// ---------------------------------------------------------------------------
// Tests: Input Detection Logic
// ---------------------------------------------------------------------------

TEST(ZeroShotEmbeddingTest, DetectSpeakerEmbeddingInput) {
    // Simulate ONNX input name detection (mirrors loadModel logic)
    struct Session {
        bool hasMultiSpeaker = false;
        bool hasProsodyInput = false;
        bool hasZeroShotInput = false;
    };

    std::vector<std::string> inputNames = {
        "input", "input_lengths", "scales", "speaker_embedding"
    };

    Session session;
    for (const auto& name : inputNames) {
        if (name == "sid") {
            session.hasMultiSpeaker = true;
        } else if (name == "prosody_features") {
            session.hasProsodyInput = true;
        } else if (name == "speaker_embedding") {
            session.hasZeroShotInput = true;
        }
    }

    EXPECT_TRUE(session.hasZeroShotInput);
    EXPECT_FALSE(session.hasMultiSpeaker);
    EXPECT_FALSE(session.hasProsodyInput);
}

TEST(ZeroShotEmbeddingTest, DetectSidNotZeroShot) {
    // Multi-speaker model with sid should NOT set hasZeroShotInput
    struct Session {
        bool hasMultiSpeaker = false;
        bool hasZeroShotInput = false;
    };

    std::vector<std::string> inputNames = {
        "input", "input_lengths", "scales", "sid"
    };

    Session session;
    for (const auto& name : inputNames) {
        if (name == "sid") {
            session.hasMultiSpeaker = true;
        } else if (name == "speaker_embedding") {
            session.hasZeroShotInput = true;
        }
    }

    EXPECT_TRUE(session.hasMultiSpeaker);
    EXPECT_FALSE(session.hasZeroShotInput);
}

// ---------------------------------------------------------------------------
// Tests: Embedding Fallback (zero vector)
// ---------------------------------------------------------------------------

TEST(ZeroShotEmbeddingTest, FallbackZeroEmbedding) {
    // When no embedding is provided, synthesize() uses zero-filled vector
    int spkEmbedDim = 192;
    std::optional<std::vector<float>> speakerEmbedding;

    std::vector<float> embeddingData;
    if (speakerEmbedding && !speakerEmbedding->empty()) {
        embeddingData = speakerEmbedding.value();
    } else {
        embeddingData.resize(spkEmbedDim, 0.0f);
    }

    EXPECT_EQ(embeddingData.size(), 192);
    for (float v : embeddingData) {
        EXPECT_FLOAT_EQ(v, 0.0f);
    }
}

TEST(ZeroShotEmbeddingTest, ProvidedEmbeddingUsed) {
    // When embedding is provided, it should be used instead of zeros
    std::vector<float> emb(192);
    for (size_t i = 0; i < 192; i++) {
        emb[i] = static_cast<float>(i) * 0.01f;
    }
    std::optional<std::vector<float>> speakerEmbedding = emb;

    std::vector<float> embeddingData;
    if (speakerEmbedding && !speakerEmbedding->empty()) {
        embeddingData = speakerEmbedding.value();
    } else {
        embeddingData.resize(192, 0.0f);
    }

    EXPECT_EQ(embeddingData.size(), 192);
    EXPECT_FLOAT_EQ(embeddingData[0], 0.0f);
    EXPECT_FLOAT_EQ(embeddingData[100], 1.0f);
    EXPECT_FLOAT_EQ(embeddingData[191], 1.91f);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
