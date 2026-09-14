# Benchmark papers

- Title: VERA-MH: Validation of Ethical and Responsible AI in Mental Health
- Authors: Luca Belli, Kate H. Bentley, Josh Gieringer, Emily Van Ark, Nilu Zhao, Pradip Thachile, Matt Hawrilenko, Millard Brown, Adam M. Chekroud (Spring Health)
- Canonical source: https://arxiv.org/abs/2605.13318 (submitted 13 May 2026; the repository's CITATION.cff points here)
- Immutable PDF: https://arxiv.org/pdf/2605.13318
- SHA-256 of the PDF as fetched on 2026-09-14: `eab472d0847c13aebfb539ff8c7fc1381bbb51e80f16414d4d003b16abe7912d`
- Related: reliability/validity study https://arxiv.org/abs/2602.05088; concept paper https://arxiv.org/abs/2510.15297
- Code and data: https://github.com/SpringCare/VERA-MH at `2c9d1fcbb68e1a2df64171c18b3e4d4c18b2f89e` (rubric v1.2; license in `resources_servers/vera_mh/rubric/LICENSE`)

The PDF is not committed. `python benchmarks/vera_mh/fetch_paper.py` downloads it into this directory (gitignored)
and verifies the SHA-256; run packages copy the verified file under `paper/`. arXiv does not pin a version suffix
for this listing, so the hash is the fetch-time pin; a mismatch means arXiv published a revision.
