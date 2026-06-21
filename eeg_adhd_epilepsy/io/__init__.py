"""EEG I/O sub-package.

Public surface
--------------
- ``bids``              — BIDS path/naming contract, discovery, stage layout
- ``ingest``            — raw recording discovery and ``.pnt`` parsing
- ``recording``         — recording-level grouping helpers
- ``patients``          — canonical patient-metadata builder (eeg-build-patients-metadata)
- ``containers``        — analysis input loading/shaping into coco-pipe DataContainers
- ``descriptor_layout`` — file-layout contract for descriptor shards
"""
