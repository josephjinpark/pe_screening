# PE_Analysis

Analysis platform for prime editing guide RNA (pegRNA) library analysis for studies entitled, "Predicting the Efficiency of Prime Editing Guide RNAs in Human Cells", Nat. Biotech. 2021, https://doi.org/10.1038/s41587-020-0677-y and "Prediction of efficiencies for diverse prime editing systems in multiple cell types", Cell 2023, https://doi.org/10.1016/j.cell.2023.03.034

A_main.py : Prime editing guide RNA (pegRNA) design.
- Load, parse, and filter the ClinVar database, https://www.ncbi.nlm.nih.gov/clinvar/, for human pathogenic variants that elicit 1-3bp sized insertions, deletions, or substitutions.
- Survey all possible variants that can be targeted by NGG PAM sequence.
- Determine all possible reverse-transcriptase template (RTT) and primer-binding site (PBS) sequences for pegRNA design.

B_main.py : Prime editing NGS analysis from library screening experiments.
- Load and sort NGS data from pegRNA according to barcode.
- Categorize NGS reads according to outcome, WT, PE, or others. 
- Output for follow-up analyses.

C.main.py : Feature extraction pipeline for model input.
- Determine sequence context and properties for pegRNA components (melting temperature, GC content, and minimum free energies) using BioPython and ViennaRNA packages.
