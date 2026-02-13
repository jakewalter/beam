# Makefile for typical pipeline tasks
.PHONY: all detect triage cluster test lint format

PY=python3
PYTHONPATH=.

all: detect triage cluster

detect:
	PYTHONPATH=$(PYTHONPATH) $(PY) beam_driver.py --mode traditional --data-dir /path/to/data --start 20200601 --end 20200601 --plot --plot-dir ./plots

triage:
	PYTHONPATH=$(PYTHONPATH) $(PY) scripts/triangulate_from_detections_json.py --detections plots/detections_20200601.json --centers plots/centers.json --date 20200601 --time-tol 30.0 --outdir plots

cluster:
	PYTHONPATH=$(PYTHONPATH) $(PY) scripts/cluster_locations.py --locations plots/locations_20200601.json --out plots/locations_summary_20200601.csv --cluster-km 20.0 --min-members 1

test:
	PYTHONPATH=$(PYTHONPATH) $(PY) -m pytest -q

format:
	black .
	isort .

lint:
	flake8
