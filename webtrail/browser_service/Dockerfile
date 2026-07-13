# Browser service: Playwright Chromium + the typed-action HTTP server.
# The official Playwright image already ships the browser, its system libs, and
# base fonts; we add CJK/emoji fonts so non-Latin sites don't render as tofu.
FROM mcr.microsoft.com/playwright:v1.47.0-jammy

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl fonts-noto-cjk fonts-noto-color-emoji \
 && rm -rf /var/lib/apt/lists/*

COPY package.json package-lock.json* ./
RUN npm install --omit=dev

COPY src ./src
COPY start.sh ./
RUN chmod +x start.sh

# Chromium cannot run its sandbox inside most containers; server.js already
# disables it on Linux unless WEBTRAIL_SANDBOX=1.
ENV WORKERS=4 BASE_PORT=9300
EXPOSE 9300-9307

CMD ["bash", "start.sh"]
