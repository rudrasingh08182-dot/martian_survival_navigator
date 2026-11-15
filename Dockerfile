
# Martian Map - FastAPI backend + frontend in one container
# Designed for Render, Google Cloud Run and Azure Container Apps.

FROM python:3.12-slim

# --------------------------------------------------
# 1. Install system dependencies required by Rasterio
# --------------------------------------------------
RUN apt-get update \
    && apt-get install -y --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------
# 2. Create a non-root application user
# --------------------------------------------------
RUN useradd -m -u 1000 user

USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /home/user/app

# --------------------------------------------------
# 3. Install Python dependencies
# --------------------------------------------------
COPY --chown=user requirements.txt .

RUN pip install --no-cache-dir --user -r requirements.txt

# --------------------------------------------------
# 4. Copy complete project
# --------------------------------------------------
COPY --chown=user . .

# --------------------------------------------------
# 5. Extract NASA/PSI SWIM 2.0 datasets
# --------------------------------------------------
# SWIMProducts.zip must be present in the project root.

RUN mkdir -p data/swim \
    && python -c "from pathlib import Path; from zipfile import ZipFile; archive = Path('SWIMProducts.zip'); assert archive.is_file(), 'SWIMProducts.zip is missing from the Docker build context'; ZipFile(archive).extractall('data/swim')" \
    && python -c "from pathlib import Path; required = ['SWIM2_c0_1.tif', 'SWIM2_c1_5.tif', 'SWIM2_c_5.tif']; missing = [name for name in required if not (Path('data/swim') / name).is_file()]; assert not missing, f'Missing SWIM datasets: {missing}'"

# --------------------------------------------------
# 6. Download elevation datasets
# --------------------------------------------------
# MOLA, Jezero CTX DTM and Gale HiRISE DTM.

RUN python scripts/fetch_data.py

# --------------------------------------------------
# 7. Render web service port
# --------------------------------------------------
EXPOSE 10000

# --------------------------------------------------
# 8. Start FastAPI
# --------------------------------------------------
CMD uvicorn app:app \
    --host 0.0.0.0 \
    --port ${PORT:-10000} \
    --app-dir backend \
    --proxy-headers \
    --forwarded-allow-ips "*"
