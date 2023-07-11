#!/bin/bash

pytest --cov=atomlib
pytest --cov=atomlib --cov-append --nbmake examples/*.ipynb
coveragepy-lcov
