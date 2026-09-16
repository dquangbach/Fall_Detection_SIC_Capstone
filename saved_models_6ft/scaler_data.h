#ifndef SCALER_DATA_H_
#define SCALER_DATA_H_

#include <cmath>
#include <cstdint>

constexpr int kWindowSize = 200;
constexpr int kWindowStep = 50;
constexpr int kNumFeatures = 6;
constexpr int kInputSize = 1200;
constexpr int kScalerSize = 6;
// Dequantized probability >= kFallThreshold -> Fall.
// Keep the Python threshold as double to preserve its comparison semantics.
constexpr double kFallThreshold = 0.5;

// Flattened window: [AccX, AccY, AccZ, GyrX, GyrY, GyrZ] for each time step, oldest first.
constexpr float kScalerMean[kScalerSize] = {-9.556641453e-04f, -9.180242419e-01f, -7.702551782e-02f, -5.174291134e+00f, 8.428470612e+00f, -7.061172724e-01f};
constexpr float kScalerScale[kScalerSize] = {2.383738309e-01f, 3.519431353e-01f, 3.517512381e-01f, 4.133506393e+01f, 4.697683334e+01f, 2.058464432e+01f};

constexpr float kInputScale = 1.298178732e-01f;
constexpr int kInputZeroPoint = -5;
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
