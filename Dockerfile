# Pinned image and model-specific compatibility patch; see README for sources.
FROM lmsysorg/sglang:inkling-2-mini@sha256:efac853b46811ccb760545b7b29e7119bdf907537ccf6e863fac3011fe8b98de
COPY scripts/sglang_inkling_dflash2.py /tmp/sglang_inkling_dflash2.py
RUN python /tmp/sglang_inkling_dflash2.py && rm /tmp/sglang_inkling_dflash2.py
