"""Registry of every text source the corpus is built from.

Each entry names where the file comes from and under what terms, so
DATA_LICENSES.md and the build stay in step.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    key: str
    url: str
    filename: str
    license: str


SOURCES = {
    s.key: s
    for s in [
        Source(
            key="kjv",
            url="https://www.gutenberg.org/cache/epub/10/pg10.txt",
            filename="pg10_kjv.txt",
            license="Public domain (Project Gutenberg #10; header and footer stripped)",
        ),
        Source(
            key="webster",
            url="https://www.gutenberg.org/cache/epub/29765/pg29765.txt",
            filename="pg29765_webster1913.txt",
            license="Public domain (Project Gutenberg #29765; header and footer stripped)",
        ),
        Source(
            key="catalog",
            url="https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv",
            filename="pg_catalog.csv",
            license="Project Gutenberg catalog metadata; titles of public-domain works",
        ),
        Source(
            key="surnames",
            url="https://www2.census.gov/topics/genealogy/2010surnames/names.zip",
            filename="census2010_surnames.zip",
            license="US Census Bureau, US government work, public domain",
        ),
        Source(
            key="first_female",
            url="https://www2.census.gov/topics/genealogy/1990surnames/dist.female.first",
            filename="census1990_female_first.txt",
            license="US Census Bureau, US government work, public domain",
        ),
        Source(
            key="first_male",
            url="https://www2.census.gov/topics/genealogy/1990surnames/dist.male.first",
            filename="census1990_male_first.txt",
            license="US Census Bureau, US government work, public domain",
        ),
    ]
}
