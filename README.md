# Code for "Variable Ice-Rock Coupling Near the Grounding Zone of Totten Glacier, East Antarctica, Inferred From Repeating Stick-Slip Icequakes"
Jared C. Magyar, 2026

### Background

The scripts in this repository can be used to reproduce the results in "Variable Ice-Rock Coupling Near the Grounding Zone of Totten Glacier, East Antarctica, Inferred From Repeating Stick-Slip Icequakes". It also provides an example end-to-end workflow for using the 'cryoquake' analysis tools, currently under development, and can be found at github.com/JMagyar15/cryoquake.

### Instructions

The scripts and notebooks included here require installation of the 'cryoquake' workflow, available at github.com/JMagyar15/cryoquake. This is under development, but can be installed by cloning/downloading the respository and using ``pip install -e /path/to/cryoquake``. This will eventually be replaced with a more stable installation with included dependencies...stay tuned!

Before running any of the scripts, the Totten Glacier data must be downloaded. This can be accessed using the Obspy MassDownloader, or using the ``download_waveforms.py`` script included here. 

The event detection catalogues are included in the repository, so any other script can be run without needing to re-do the event detection. However, the event detection can be re-run using ``event_detection.py``.


Order of running scripts should be:
1) download_waveforms.py
2) event_detection.py
3) compute_attributes.py (OPTIONAL)
4) template_matching.py
5) location_inversion.py
6) moment_magnitude.py