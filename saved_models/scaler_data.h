#ifndef SCALER_DATA_H_
#define SCALER_DATA_H_

#include <cstddef>

// ============================================================
// FALL DETECTION CONFIGURATION
// ============================================================

constexpr int kWindowSize = 200;
constexpr int kWindowStep = 100;
constexpr int kNumFeatures = 8;
constexpr int kInputSize = 8;

// Probability >= kFallThreshold -> Fall
constexpr float kFallThreshold = 0.55f;


// ============================================================
// FEATURE ORDER
// ============================================================

//   0: AccX
//   1: AccY
//   2: AccZ
//   3: AccMag
//   4: GyrX
//   5: GyrY
//   6: GyrZ
//   7: GyrMag


// ============================================================
// STANDARD SCALER
//
// Python preprocessing:
//     x_scaled[i] =
//         (x[i] - scaler.mean_[i])
//         / scaler.scale_[i];
//
// Firmware phải thực hiện chính xác cùng phép biến đổi.
// ============================================================

constexpr float kScalerMean[8] = {
    -0.000955664145f, -0.918024242f, -0.0770255178f, 1.0197612f, -5.17429113f, 8.42847061f,
    -0.706117272f, 39.8188744f
};


constexpr float kScalerScale[8] = {
    0.238373831f, 0.351943135f, 0.351751238f, 0.336458623f, 41.3350639f, 46.9768333f,
    20.5846443f, 53.403244f
};


// ============================================================
// SCALING HELPER
// ============================================================

inline float StandardizeInput(
    float value,
    int index
) {
    return (
        value - kScalerMean[index]
    ) / kScalerScale[index];
}

#endif  // SCALER_DATA_H_
