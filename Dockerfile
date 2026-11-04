FROM ubuntu:24.04 AS release
ADD https://git.reseam.app/reseam/translate/releases/download/latest/reseam-translate-linux-x64.tar.gz /release.tar.gz
RUN mkdir /release && tar -xzf /release.tar.gz -C /release

FROM ubuntu:24.04
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --home-dir /models translate \
    && install -d -o translate /models
COPY --from=release /release /
USER translate
ENV MODELS_DIR=/models LANGUAGES=es,fr,de,it,pt,ru,uk,pl,nl,tr,ar,fa,hi,ja,ko,zh,id,vi PORT=8080 PRELOAD=1
VOLUME /models
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=10m \
  CMD python3 -c "import urllib.request,os; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ['PORT'], timeout=2)"
ENTRYPOINT ["/usr/local/bin/entrypoint"]
