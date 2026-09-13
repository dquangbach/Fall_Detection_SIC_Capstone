#ifndef SCALER_DATA_H_
#define SCALER_DATA_H_

#include <cmath>
#include <cstdint>

constexpr int kWindowSize = 200;
constexpr int kWindowStep = 100;
constexpr int kNumFeatures = 3;
constexpr int kInputSize = 600;
constexpr int kScalerSize = 3;
// Dequantized probability >= kFallThreshold -> Fall.
// Keep the Python threshold as double to preserve its comparison semantics.
constexpr double kFallThreshold = 0.20000000000000004;

// Flattened window: [AccX, AccY, AccZ] for each time step, oldest first.
constexpr float kScalerMean[kScalerSize] = {-3.882899880e-03f, -9.565066695e-01f, -7.179296762e-02f};
constexpr float kScalerScale[kScalerSize] = {1.438738406e-01f, 2.558356524e-01f, 2.814784646e-01f};

constexpr float kInputScale = 1.932223737e-01f;
constexpr int kInputZeroPoint = -22;
constexpr float kOutputScale = 3.906250000e-03f;
constexpr int kOutputZeroPoint = -128;

inline float StandardizeInput(float value, int index) {
    const int feature_index = index % kNumFeatures;
    return (value - kScalerMean[feature_index]) / kScalerScale[feature_index];
}

inline int8_t QuantizeScaledInput(float scaled_value) {
    // nearbyint uses round-to-nearest-even by default, matching numpy.rint.
    const float quantized = std::nearbyint(scaled_value / kInputScale) + kInputZeroPoint;
    const float clipped = quantized < -128.0f ? -128.0f : (quantized > 127.0f ? 127.0f : quantized);
    return static_cast<int8_t>(clipped);
}

inline int8_t QuantizeInput(float raw_value, int flattened_index) {
    return QuantizeScaledInput(StandardizeInput(raw_value, flattened_index));
}

inline float DequantizeOutput(int8_t quantized_value) {
    return (static_cast<int>(quantized_value) - kOutputZeroPoint) * kOutputScale;
}

// Firmware: input->data.int8[i] = QuantizeInput(raw_window[i], i);
// float probability = DequantizeOutput(output->data.int8[0]);
// bool fall = probability >= kFallThreshold;
#endif  // SCALER_DATA_H_
