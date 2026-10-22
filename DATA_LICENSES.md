# Data licenses

Every text source and font used to build training data is listed here.

## Text sources

| Source | License | URL |
| --- | --- | --- |
| King James Bible (Project Gutenberg) | Public domain (Gutenberg header/footer stripped) | https://www.gutenberg.org/ebooks/10 |
| Webster's Unabridged Dictionary, 1913 (Project Gutenberg) | Public domain | https://www.gutenberg.org/ebooks/29765 |
| US Census 2010 surname list | US government work, public domain | https://www2.census.gov/topics/genealogy/2010surnames/names.zip |
| Project Gutenberg catalog (titles of public-domain genre fiction) | Catalog metadata; the works themselves are public domain | https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv |
| US Census 1990 first-name lists | US government work, public domain | https://www2.census.gov/topics/genealogy/1990surnames/ |

## Fonts

264 families (356 faces) from the Google Fonts repository, https://github.com/google/fonts. The curated list and its style tags are in `src/logo_wizard/fonts/catalog.py`; `logo-wizard-fonts` downloads them into `local/fonts/` (not committed) together with each family's license file, and writes `data/fonts/manifest.csv` with family, file, license, source URL and sha256 for every face.

| License | Faces |
| --- | --- |
| SIL Open Font License 1.1 | 324 |
| Apache License 2.0 | 27 |
| Ubuntu Font License 1.0 | 5 |

Only the `ofl/`, `apache/` and `ufl/` folders are used, and the license named in each family's METADATA.pb must match its folder. These licenses allow use and redistribution, and images drawn with the fonts carry no font-license obligations.

## Palettes

`data/palettes/palettes.yaml` is original to this project.

The exact text download URLs live in `src/logo_wizard/corpus/sources.py`.
