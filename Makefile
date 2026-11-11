# `make all` reruns everything from the official data (hours). `make figures` rebuilds
# tables/figures from committed results (minutes). Windows without make: python reproduce.py <target>
PY ?= python
RUN = $(PY) -W ignore -m
ZIP_URL = http://web.archive.org/web/20221016205142id_/http://s3-us-west-2.amazonaws.com/telemanom/data.zip
ZIP_SHA = b4d66deb492d9b0a353b51879152687ed9313897e8e19320d2dc853d738ed8a7

.PHONY: all data test train faults export figures

all: data test train faults export figures

data:
	$(RUN) scripts.get_data
	$(RUN) scripts.inspect_data

test:
	$(PY) -m pytest tests -q

train:
	$(RUN) scripts.run_detection --phase fast
	$(RUN) scripts.run_detection --phase lstm

faults:
	$(RUN) scripts.run_faults --only clean bitpos sweep transient protect

export:
	$(RUN) scripts.export_c
	$(RUN) scripts.check_c_host
	$(RUN) scripts.resources

figures:
	$(RUN) scripts.summarize_detection
	$(RUN) scripts.summarize_faults
