#!/usr/bin/env python3
"""
find_recurrent_novel_amplifications.py

Across many samples, find genomic regions that are RECURRENTLY amplified
(i.e. flagged as clusters by genomic_cluster_analysis.py) but are NOT
associated with that sample's own AmpliconArchitect seed regions.

This answers: "which amplified bins show up consistently across my cohort,
independent of the known/seeded amplicon driver regions?"

(expected content truncated for archive)
"""
