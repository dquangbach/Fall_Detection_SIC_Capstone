#ifndef SCALER_DATA_H_
#define SCALER_DATA_H_

#include <cstddef>

// ============================================================
// FALL DETECTION CONFIGURATION
// ============================================================

constexpr int kWindowSize = 200;
constexpr int kWindowStep = 100;
constexpr int kNumFeatures = 3;
constexpr int kInputSize = 3;

// Probability >= kFallThreshold -> Fall
constexpr float kFallThreshold = 0.5f;


// ============================================================
// FEATURE ORDER
// ============================================================

//   0: AccX
//   1: AccY
//   2: AccZ


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

constexpr float kScalerMean[3] = {
    -0.00388289988f, -0.95650667f, -0.0717929676f
};


constexpr float kScalerScale[3] = {
    0.143873841f, 0.255835652f, 0.281478465f
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
