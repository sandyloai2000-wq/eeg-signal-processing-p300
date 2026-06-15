# eeg-signal-processing-p300
EEG signal processing project analyzing P300 paradigm using Python, including filtering, trial extraction, and time-domain analysis.

This project focuses on processing and analyzing EEG signals in a P300 experimental paradigm using Python.

## Overview
The P300 signal is a well-known event-related potential (ERP) used in neuroscience and brain-computer interface (BCI) systems. This project extracts and analyzes EEG responses to visual stimuli.

## Features
- Loading EEG data from MATLAB (.mat) files
- Signal preprocessing (filtering and detrending)
- Detection of stimulus events
- Extraction of individual trials from continuous EEG data
- Time-domain visualization of EEG signals

## Methodology
- Identified stimulus onset using stimulus codes
- Segmented EEG signals into trials (~667 ms windows)
- Focused analysis on Cz channel (central electrode)
- Visualized signal responses over time

## Technologies
- Python
- NumPy
- SciPy
- Matplotlib

## What I learned
- EEG signal processing fundamentals
- Event-related potential (P300) analysis
- Working with biomedical datasets
- Signal segmentation and preprocessing

## Applications
- Brain-computer interfaces (BCI)
- Cognitive neuroscience research
- AI-based neural signal analysis

## Future Improvements
- Apply machine learning for classification (target vs non-target)
- Feature extraction for P300 detection
- Integration with deep learning models
