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

Recorded per font in `data/fonts/manifest.csv` (family, license, source URL, sha256). Only SIL OFL 1.1, Apache 2.0 and Ubuntu Font License fonts are accepted.

The exact download URLs live in `src/logo_wizard/corpus/sources.py`.
