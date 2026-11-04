#!/bin/sh
# Fetches the configured models (skipping files already present and verified), then serves.
set -e
if [ -n "$LANGUAGES" ]; then
  python3 /usr/local/bin/fetch-models "$MODELS_DIR" "$LANGUAGES"
fi
exec /usr/local/bin/reseam-translate "$@"
