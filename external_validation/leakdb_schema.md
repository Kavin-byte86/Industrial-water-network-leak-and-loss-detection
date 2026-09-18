# LeakDB Dataset Schema

## Overview
LeakDB (Leakage Diagnosis Benchmark) provides artificially created realistic leakage scenarios for water distribution networks (like Hanoi and Net1). The data is organized strictly by **Scenarios**.

## Scenario Structure
Each network (e.g., `Hanoi_CMH`) contains multiple scenarios (e.g., `Scenario-1`, `Scenario-2`... `Scenario-10`). 
A scenario represents an independent continuous simulation (typically 8760 hours / 1 year at 30-min intervals).

Inside a `Scenario-X` folder, the data is split across several directories and files:

### 1. `Flows/`
Contains one CSV file per pipe (e.g., `Link_1.csv`, `Link_2.csv`).
**Columns**:
- `Timestamp` (format: `YYYY-MM-DD HH:MM:SS`)
- `Value` (Flow rate passing through the link, typically in CMH - Cubic Meters per Hour)

### 2. `Pressures/`
Contains one CSV file per junction/node (e.g., `Node_1.csv`, `Node_2.csv`).
**Columns**:
- `Timestamp`
- `Value` (Pressure at the node, typically in meters of head)

### 3. `Demands/`
Contains one CSV file per junction/node representing the legitimate water consumption at that node.
**Columns**:
- `Timestamp`
- `Value` (Water demand in CMH)

### 4. `Labels.csv`
A single master file indicating if **any** leak is occurring in the network at a given timestamp.
**Columns**:
- `Timestamp`
- `Label` (`0.0` = Normal, `1.0` = Leak present)

### 5. `Leaks/` (Present only if scenario contains leaks)
Contains metadata and flow rates for specific leaks.
- `Leak_{NodeID}_info.csv`: Contains metadata like Leak Node, Area, Diameter, Type (abrupt/incipient), Start Time, End Time.
- `Leak_{NodeID}_demand.csv`: Time-series of the exact water lost to this specific leak (the actual leak rate).

### 6. `{Network}_Scenario-{X}.inp`
The official EPANET topology file defining how nodes (junctions) and links (pipes) are connected. This is required to dynamically construct Network-Balance features.
