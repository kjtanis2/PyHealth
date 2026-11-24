pyhealth.tasks.SleepApneaDetection
==================================

Sleep apnea detection on 30 second windows of pediatric sleep studies, for
:class:`~pyhealth.datasets.NCHSleepDataset` and
:class:`~pyhealth.datasets.CHATDataset`.

Each window is labelled 1 when an apnea or hypopnea was scored during it, and
windows scored as wake are left out. The signal holds one channel per
modality (EOG, EEG, airflow, SpO2, CO2) and two channels derived from the ECG
(R-R interval and R-peak amplitude), so a model can be trained or tested on a
subset of the modalities, or with a modality filled with zeros
(``fill_missing=True``) to simulate a missing one. This follows the data
pipeline of the apnea-missing-modality code
(https://github.com/healthylaife/apnea-missing-modality).

Studies of one patient are related, so split by patient with
``split_by_patient``.

Task parameters:

- modalities: subset of ``("eog", "eeg", "resp", "spo2", "co2", "ecg")``,
  default all
- sampling_rate: float, default=64
- window_seconds: int, default=30
- min_ahi: float or None, default=None (the original code uses 3 for NCH)
- fill_missing: bool, default=False

Differences from the original code
----------------------------------

- R peaks are found with ``scipy.signal.find_peaks`` after a band-pass
  filter, instead of biosppy's Hamilton segmenter.
- Every window is kept; the original drops negative windows at random to
  balance the classes.
- Voltages are in microvolts and the R-peak amplitude is not clipped; the
  original keeps volts, clips the amplitude, and rescales the CHAT R-R
  interval and amplitude.

.. autoclass:: pyhealth.tasks.SleepApneaDetection
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.tasks.sleep_apnea_detection.detect_r_peaks

.. autofunction:: pyhealth.tasks.sleep_apnea_detection.ecg_features
