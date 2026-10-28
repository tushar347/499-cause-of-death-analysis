# Data access

## Data is not included

This repository contains no NMDID person-level records, feature matrix, or CT images. The local `data/` directory is ignored by Git except for this guidance file. Do not upload source data, derived person-level records, identifiers, or images to the repository, issues, pull requests, or public reports.

## Requesting access

1. Visit the [New Mexico Decedent Image Database](https://nmdid.unm.edu) and review its current access requirements and terms.
2. Submit the required researcher/account and project information, including institutional or ethics approvals where applicable.
3. Wait for authorization before searching, viewing, or downloading restricted records.
4. Use the database search interface to identify eligible records. The search interface has a 500-record limit per search; plan searches accordingly and do not treat that limit as permission to bypass access controls.
5. Add eligible records to carts and follow the NMDID workflow for reviewing and requesting the selected records.
6. Store approved files only in access-controlled locations, follow the database terms for use and retention, and keep the repository copy limited to aggregate, non-identifying outputs.

Database workflows and requirements can change; consult the NMDID website and your approved access terms for current instructions. Access does not authorize redistribution or public release.

## Local use

For the current metadata scripts, place the authorized local feature matrix at `data/metadata_feature_matrix.csv`. The filename is explicitly ignored. Keep DICOM/NIfTI data and other source exports out of Git as well.

## Reference

Berry and Edgar, “Announcement: The New Mexico Decedent Image Database,” *Forensic Imaging*, vol. 24, 200436, 2021. https://doi.org/10.1016/j.fri.2021.200436
