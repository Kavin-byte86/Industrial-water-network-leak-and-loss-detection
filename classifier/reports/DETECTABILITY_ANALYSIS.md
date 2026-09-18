# Sensor Noise & Detectability Analysis

## Noise Floor Profile
- Configured Flow Noise: 2.0%
- Typical Main Inlet Flow (J1): ~1000.0 L/min
- Expected Absolute Noise Floor: ±20.0 L/min

## Detectability Conclusion
A synthetic leak of 3-5 L/min represents ~0.3% of the total network flow. This is structurally buried beneath the 2.0% measurement noise. 
We MUST NOT force the ML model to blindly trigger on mathematically imperceptible signals, as this would drastically increase false alarms during normal legitimate demand variations.

Minimum practically detectable leak (Signal-to-Noise Ratio > 1.5): **~30.0 L/min**
