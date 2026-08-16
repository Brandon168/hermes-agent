# Vercel AI Gateway v4 protocol oracle

This is a **development-only** conformance probe. Hermes production remains
Python-only; Node and the Vercel SDK are not runtime dependencies.

The package versions are exact. When upgrading any version:

1. Run `npm ci` in this directory.
2. Export the same `AI_GATEWAY_API_KEY` used by the target named Hermes
   provider.
3. Run `npm run capture > capture.json`.
4. Compare the redacted request/response shapes with
   `tests/fixtures/vercel_ai_gateway/sdk_oracle_v4.json`.
5. Update the fixture and parser only after deterministic tests pass.
6. Run the capped Luna low-reasoning live QA before deployment.

The script removes the Authorization header from output. Treat captured search
content and Gateway metadata as potentially sensitive regardless.
