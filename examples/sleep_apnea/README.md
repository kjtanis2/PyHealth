# Mock NCH / CHAT data

Synthetic stand-ins for the NSRR NCH Sleep DataBank and CHAT sleep studies, for running
`NCHSleepDataset`, `CHATDataset` and `SleepApneaDetection` without data access.

From the PyHealth repo root, with PyHealth installed (`pip install -e .`):

```
python examples/sleep_apnea/make_mock_apnea_data.py --force                                            # out/  (20 min studies)
python examples/sleep_apnea/make_mock_apnea_data.py --minutes 6 --out examples/sleep_apnea/tiny --force # tiny/ (6 min studies, ~64 MB)
```

Output is deterministic for a given `--seed` (default 598). Both output folders are gitignored;
regenerate them locally rather than committing them.

## Layout

| Path | Matches |
|---|---|
| `raw/nchsdb/sleep_data/<pat>_<study>.edf` + `.tsv` (onset, duration, description) | NCH download |
| `raw/nchsdb/health_data/SLEEP_STUDY.csv`, `DEMOGRAPHIC.csv` | NCH health data (subset of columns) |
| `raw/chat/polysomnography/edfs/{baseline,followup}/chat-<visit>-<id>.edf` | CHAT download |
| `raw/chat/polysomnography/annotations-events-nsrr/<visit>/chat-<visit>-<id>-nsrr.xml` | NSRR XML annotations |
| `raw/chat/polysomnography/edfs/baseline/chat-baseline-<id>-nsrr.tsv` | the TSV apnea-missing-modality reads (baseline only) |
| `raw/chat/datasets/chat-{baseline,followup}-dataset-mock.csv` | CHAT subject tables (subset of columns) |

`manifest.json` lists the events placed in every study and its AHI.

## Loading it

```python
from pyhealth.datasets import CHATDataset, NCHSleepDataset
from pyhealth.tasks import SleepApneaDetection

if __name__ == "__main__":  # needed on Windows: PyHealth starts worker processes
    nch = NCHSleepDataset(root="examples/sleep_apnea/out/raw/nchsdb")
    nch.stats()   # 8 patients, 10 studies
    samples = nch.set_task(SleepApneaDetection(min_ahi=3))
    print(samples[0]["signal"].shape)  # (7, 1920): 30 s at 64 Hz, 7 channels

    chat = CHATDataset(root="examples/sleep_apnea/out/raw/chat")
    chat.stats()  # 8 patients, 10 studies (8 baseline + 2 follow-up)
```

## Edge cases included on purpose
- NCH patients 10012 and 10071 each have two studies, and CHAT 300001 and 300002 a
  follow-up visit (XML annotations only), so splits must be by patient.
- NCH 10063 is recorded at 400 Hz and CHAT 300003 at 256 Hz, so resampling is exercised.
- NCH 10104 and CHAT 300007 have no events (AHI 0): `min_ahi=3` leaves them out.
- NCH 10117 has no `Capno` channel and CHAT 300008 no `Cap` channel: the task skips them,
  or fills the CO2 modality with zeros with `fill_missing=True`.
- Every study uses one sampling rate for all channels. Real NSRR EDFs often mix rates
  (for example, SpO2 at a lower rate); mne resamples those to the highest rate on reading.

Window counts, labels and event seconds from `SleepApneaDetection` were checked against a port
of the original `preprocessing.py` on this data and agree for every study.
