FROM python:3.8

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    python3-distutils \
    libopenblas-dev \
    liblapack-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install a modern Rust toolchain for river builds on linux/aarch64.
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain 1.75.0

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install "setuptools<60" wheel

# scikit-multiflow checks for NumPy during setup.py metadata generation.
RUN pip install --no-cache-dir "numpy==1.24.4" "Cython==3.2.3"

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py", "--help"]