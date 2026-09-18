# LeakDB Unit Documentation

Based on inspection of the dataset metadata, EPANET `.inp` files (specifically `Hanoi_CMH`), and the LeakDB benchmark documentation, the physical units are as follows:

| Measurement | Unit | Notes |
| :--- | :--- | :--- |
| **Flow** | CMH (Cubic Meters per Hour) | Indicated by the `_CMH` suffix in the benchmark directory `Hanoi_CMH`. |
| **Pressure** | Meters of Head (m) | Standard EPANET metric unit for pressure/head. |
| **Demand** | CMH (Cubic Meters per Hour) | Consistent with flow units. |
| **Leak Rate / Magnitude** | CMH (Cubic Meters per Hour) | Reported in the `Leak_*_demand.csv` files representing physical volume lost. |
| **Pipe Length** | Meters (m) | Standard EPANET metric units. |
| **Pipe Diameter** | Millimeters (mm) | Standard EPANET metric units. |
| **Sampling Interval** | 30 Minutes | Data is reported at 30-minute intervals (`YYYY-MM-DD HH:00:00` and `YYYY-MM-DD HH:30:00`). |

**Interpretation regarding Textile Model Transferability:**
The textile ML model expects flow inputs. If the textile data was generated in L/min, the magnitudes will be completely different from CMH (1 CMH = 16.67 L/min). This further reinforces why external validation of the *physical methodology* (Residual + Balance) is more meaningful than direct model weights transfer.
