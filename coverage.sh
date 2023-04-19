#!/bin/bash

pytest --cov=structlib
pytest --cov=structlib --cov-append --nbmake examples/*.ipynb
coveragepy-lcov
