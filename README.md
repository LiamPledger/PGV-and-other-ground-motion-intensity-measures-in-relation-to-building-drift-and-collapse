# PGV and Other Ground Motion Intensity Measures in Relation to Building Drift and Collapse

## Overview

This repository contains all data, Python scripts, ground motion records, and building models used in the study "_PGV and other ground motion intensity measures in relation to building drift and collapse_". The goal is to enable full transparency and reproducibility for researchers and anyone interested in using the data.

## Contents


- **Ground Motions and IMs**: .txt files of the 100 ground motion time-history records used for NLTHA. Also includes python scripts and excel files used to compute the various intensity measures.
- **Building Models**: Python scripts of the numerical models used in the study. Includes the code to conduct IDA, as well as all of the models and information about each building (i.e. section sizes, reinforecment ratios...)
- **IDA Results**: Data used to develop IDA curves are provided here. Excel files for each model (max_drift.xlsx) include peak storey drift recorded for each increment of spectral acceleration used in IDA for all 100 ground motions. 
- **SaRatio and Ds plots**: Python scripts and relevant GM data to reproduce Figure 5 and 6.
- **Sufficiency plots**: Python scripts and relevant data for the plots illustrating the sufficiency of the different IMs.
- **Efficiency plots**: Python scripts and relevant data for the plots illustrating the sufficiency of the different IMs.


## Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/LiamPledger/PGV-and-other-ground-motion-intensity-measures-in-relation-to-building-drift-and-collapse.git
   cd PGV-and-other-ground-motion-intensity-measures-in-relation-to-building-drift-and-collapse
   ```
   
2. **Usage**:
   - Please feel free to use the python scripts for data analysis and modeling.
   - Input data, ground motions, and numerical models can be found in their respective directories.
   - Processed data of intensity measures and python scripts can be used to create new plots / investigate new IMs or trends related to the data.
   - The IDA data could also be used to evaluate IMs at a range of drift thresholds, not just collapse as was done in this study.

## Contributing

Contributions, suggestions, and questions are welcome! Please open an issue or submit a pull request.

## Contact

For questions or collaboration, please contact [LiamPledger](https://github.com/LiamPledger).

