# SPDX-FileCopyrightText: Copyright 2026 metamong
# SPDX-License-Identifier: Apache-2.0
#
# Router image for the OSSP 2026 SKT LLM routing challenge.
#
# Deviates from container/Dockerfile in one way that matters: the router needs
# scikit-learn and scipy at runtime, which have no musl wheels, so this uses the
# Debian slim base rather than the reference Alpine one. Building them from source
# on Alpine would add a toolchain to the image for no benefit. The reference base
# is a suggestion, not a requirement -- RUNTIME.md says participants may install
# public dependencies at build time provided nothing downloads at run time.
#
# Everything else follows the runtime contract: linux/arm64, no network at run
# time, read-only root, unprivileged 65532:65532, and no VOLUME declarations.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/router \
    TMPDIR=/tmp \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

# Pinned to the exact versions that fitted and pickled artifact/router.pkl.
# scikit-learn unpickles the fitted models, so version drift here either fails
# loudly or, worse, silently changes predictions. The earlier 1.5.2 pin did not
# match the 1.8.0 that produced the artifact, and every container run logged
# InconsistentVersionWarning for five estimators. The base is 3.12 rather than
# 3.11 for the same reason: numpy 2.5.1 requires >=3.12, so pinning to the
# fitting environment fixes the interpreter too.
RUN pip install --no-cache-dir --root-user-action=ignore \
        "numpy==2.5.1" "scipy==1.17.0" "scikit-learn==1.8.0" \
    && pip uninstall --yes pip setuptools wheel

COPY --chown=65532:65532 router_features.py router_run.py /opt/router/
COPY --chown=65532:65532 artifact/router.pkl /opt/router/artifact/router.pkl

# Fail the build rather than the evaluation if the artifact does not load under
# the versions actually installed here.
RUN python3 -W error::UserWarning -c "\
import pickle, pathlib; \
a = pickle.loads(pathlib.Path('/opt/router/artifact/router.pkl').read_bytes()); \
print('artifact loads:', sorted(a)[:6]); \
assert a['light'] and a['upgrades'], 'artifact missing models'"

WORKDIR /opt/router
USER 65532:65532

ENTRYPOINT ["python3", "/opt/router/router_run.py"]
