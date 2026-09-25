# On-call Runbook

This runbook covers the platform on-call rotation for the Kestrel fleet cloud.

## Severity Levels

- SEV1: fleet-wide outage or customer data at risk. Acknowledge within 5 minutes.
- SEV2: major feature degraded for many customers. Acknowledge within 15 minutes.
- SEV3: minor degradation with a workaround. Acknowledge within 4 hours.
- SEV4: cosmetic or internal-only issue. Handle during business hours.

Pages are sent through the Beacon paging app. If the primary on-call engineer does not acknowledge a SEV1 page within 5 minutes, Beacon escalates to the secondary, and after 15 minutes to the engineering manager.

## Error Codes

- E-4471: the upstream payment gateway timed out. Retry with backoff, then check the gateway's status page. Do not refund manually.
- E-5102: inventory sync lag above 10 minutes. Restart the sync worker from the admin console.
- E-3009: authentication token expired. The client should refresh its token; no action is needed on the server.
- E-7710: robot telemetry stream dropped. Check the regional MQTT broker health dashboard.

## Incident Command

For every SEV1 and SEV2 an incident commander is appointed. The incident commander owns communication, and a status update is posted every 30 minutes until resolution. A blameless postmortem is due within 5 business days after any SEV1.
