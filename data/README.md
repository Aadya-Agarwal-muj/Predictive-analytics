# Data provenance

| File | Origin | Modified? |
|---|---|---|
| `raw/SMPracticals_1.4-3.2.tar.gz` | Downloaded manually from CRAN: <https://cran.r-project.org/src/contrib/SMPracticals_1.4-3.2.tar.gz> (162,136 bytes; package dated 2025-05-27) | No |
| `raw/limits.rda` | The single member `SMPracticals/data/limits.rda`, extracted byte-for-byte by `scripts/02_extract_dataset.py` | No |
| `raw/limits_raw.csv` | `limits.rda` read with `pyreadr` and written to CSV with no cleaning | Format only |
| `processed/swedish_speed_limit_92.csv` | Output of `scripts/03_preprocess.py`: validated, typed as integers, with the added columns `log_y1` and `regime` | Yes (documented) |

## Official documentation (from `SMPracticals/man/limits.Rd`)

> The data are numbers of traffic accidents with personal injuries, reported to the police, on Swedish roads on 92 days in 1961 and 92 matching days in 1962. On some of these days a general speed limit of 90 or 100 km/hour was imposed.

* `day`: a factor indicating the day, coded 1–92.
* `lim1`: 1 indicates a limit imposed in 1961, 0 not.
* `lim2`: 1 indicates a limit imposed in 1962, 0 not.
* `y1`: number of accidents on this day in 1961.
* `y2`: number of accidents on this day in 1962.

Source: Svensson, A. (1981). On a goodness-of-fit test for multiplicative Poisson models. *Annals of Statistics*, 9, 697–704.
