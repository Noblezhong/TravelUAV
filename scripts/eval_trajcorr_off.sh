#!/bin/bash

set -euo pipefail

TRAJCORR_MODE=off exec bash "$(dirname "$0")/eval_dnn_jetson.sh"
