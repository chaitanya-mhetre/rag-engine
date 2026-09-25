# Fleet API Guide

The Fleet API lets customers manage robots, missions, and telemetry programmatically.

## Authentication

Every request must include an API key in the Authorization header as a bearer token. Live keys start with the prefix kr_live_ and test keys start with kr_test_. Keys can be created and revoked in the customer dashboard.

## Rate Limits

Each API key may make up to 600 requests per minute. Requests above the limit receive HTTP 429 with a Retry-After header. Enterprise plans can request a higher limit.

## Pagination

List endpoints use cursor-based pagination. Pass the next_cursor value from a response as the cursor parameter to fetch the next page. The maximum page size is 100 items.

## Idempotency

POST requests accept an Idempotency-Key header. Keys are remembered for 24 hours; repeating a request with the same key returns the original response instead of creating a duplicate mission.

## Webhooks

Webhook deliveries are retried up to 8 times with exponential backoff if your endpoint does not return a 2xx response. Every delivery is signed with HMAC-SHA256 and the signature is sent in the X-Kestrel-Signature header. Verify the signature before trusting the payload.
