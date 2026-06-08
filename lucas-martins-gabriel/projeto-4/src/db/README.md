# Database schema

Initial Postgres schema for the UDA pipeline.

Apply locally after starting Postgres:

```bash
psql postgresql://admin:admin@localhost:5432/uda -f src/db/001_initial_schema.sql
```

The schema stores:

- document catalog and SHA-256 idempotency keys;
- parsed semantic chunks;
- extraction runs with model/prompt metadata;
- metrics with source evidence and lineage;
- ingestion sources for polling RI pages.
