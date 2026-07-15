# webtrail CLI (collect / judge / filter / export / view / doctor / tasks).
# The browser lives in the separate browser_service image; this one only needs
# Python. Run it against the service over the compose network.
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY webtrail ./webtrail
RUN pip install --no-cache-dir -e ".[data]"

ENTRYPOINT ["python", "-m", "webtrail"]
CMD ["--help"]
