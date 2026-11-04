reseam_version := "0.28.0"

check:
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j {{ num_cpus() }}
    ctest --test-dir build --output-on-failure

release: check
    rm -rf root dist
    install -D build/reseam-translate root/usr/local/bin/reseam-translate
    install -D scripts/fetch-models.py root/usr/local/bin/fetch-models
    install -D docker/entrypoint.sh root/usr/local/bin/entrypoint
    mkdir -p dist
    tar --owner=0 --group=0 --numeric-owner -C root -czf dist/reseam-translate-linux-x64.tar.gz .
    curl -fsSL --create-dirs -o build/reseam https://cdn.reseam.app/engine/v{{ reseam_version }}/reseam-linux-x64
    chmod +x build/reseam
    git tag -f latest
    git push -f origin latest
    git log -1 --format='- %s' > build/notes.md
    build/reseam release files dist --tag latest --notes build/notes.md
    curl -fsS -X POST -H "Content-Type: application/json" -H "X-Gitea-Event: push" \
        -d '{"ref": "refs/heads/main", "after": "'"$GITHUB_SHA"'"}' "$DOKPLOY_WEBHOOK"
