CREATE EXTENSION IF NOT EXISTS pgcrypto;

DO $$
BEGIN
    CREATE TYPE company_code AS ENUM (
        'MRV',
        'CURY',
        'TENDA',
        'DIRECIONAL',
        'PLANO_PLANO',
        'PACAEMBU'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

DO $$
BEGIN
    CREATE TYPE report_type AS ENUM (
        'operational_preview',
        'earnings_release',
        'conjuntura_bulletin',
        'other'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

DO $$
BEGIN
    CREATE TYPE document_status AS ENUM (
        'discovered',
        'downloaded',
        'parsed',
        'extracted',
        'validated',
        'failed',
        'skipped_duplicate'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

DO $$
BEGIN
    CREATE TYPE extraction_strategy AS ENUM (
        'full_scan',
        'semantic_chunking',
        'hybrid'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

DO $$
BEGIN
    CREATE TYPE metric_category AS ENUM (
        'landbank',
        'launches',
        'sales',
        'transfers',
        'production',
        'cash_generation',
        'vso',
        'financing',
        'other'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

DO $$
BEGIN
    CREATE TYPE metric_unit AS ENUM (
        'BRL',
        'BRL_million',
        'BRL_billion',
        'USD',
        'USD_million',
        'units',
        'percent',
        'percentage_points',
        'thousand_BRL',
        'other'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

CREATE TABLE IF NOT EXISTS companies (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    code company_code NOT NULL UNIQUE,
    name text NOT NULL,
    ri_base_url text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS documents (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id uuid NOT NULL REFERENCES companies(id),
    report_type report_type NOT NULL DEFAULT 'operational_preview',
    year integer NOT NULL CHECK (year BETWEEN 2000 AND 2100),
    quarter integer NOT NULL CHECK (quarter BETWEEN 1 AND 4),
    period_label text GENERATED ALWAYS AS ((quarter::text || 'T' || right(year::text, 2))) STORED,
    source_url text,
    pdf_hash char(64) NOT NULL UNIQUE CHECK (pdf_hash ~ '^[0-9a-f]{64}$'),
    original_filename text,
    storage_path text,
    parsed_storage_path text,
    validated_storage_path text,
    status document_status NOT NULL DEFAULT 'discovered',
    discovered_at timestamptz,
    downloaded_at timestamptz,
    parsed_at timestamptz,
    processed_at timestamptz,
    error_message text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (company_id, report_type, year, quarter, source_url)
);

CREATE TABLE IF NOT EXISTS extraction_runs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    strategy extraction_strategy NOT NULL,
    parser text NOT NULL,
    llm_provider text NOT NULL,
    llm_model text NOT NULL,
    prompt_version text NOT NULL,
    status text NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
    input_tokens integer CHECK (input_tokens IS NULL OR input_tokens >= 0),
    output_tokens integer CHECK (output_tokens IS NULL OR output_tokens >= 0),
    raw_response_storage_path text,
    error_message text,
    started_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS document_chunks (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal integer NOT NULL CHECK (ordinal >= 0),
    heading text,
    page_start integer CHECK (page_start IS NULL OR page_start >= 1),
    page_end integer CHECK (page_end IS NULL OR page_end >= 1),
    content text NOT NULL CHECK (length(content) > 0),
    token_count integer CHECK (token_count IS NULL OR token_count >= 0),
    parser text NOT NULL DEFAULT 'docling',
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (document_id, ordinal),
    CHECK (page_start IS NULL OR page_end IS NULL OR page_end >= page_start)
);

CREATE TABLE IF NOT EXISTS metrics (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    extraction_run_id uuid REFERENCES extraction_runs(id) ON DELETE SET NULL,
    chunk_id uuid REFERENCES document_chunks(id) ON DELETE SET NULL,
    company_id uuid NOT NULL REFERENCES companies(id),
    year integer NOT NULL CHECK (year BETWEEN 2000 AND 2100),
    quarter integer NOT NULL CHECK (quarter BETWEEN 1 AND 4),
    category metric_category NOT NULL,
    metric_name text NOT NULL CHECK (length(metric_name) > 0),
    segment text,
    value numeric,
    unit metric_unit,
    currency text CHECK (currency IS NULL OR currency IN ('BRL', 'USD')),
    scale text CHECK (
        scale IS NULL OR scale IN ('unit', 'thousand', 'million', 'billion', 'percent', 'percentage_points')
    ),
    is_estimated boolean NOT NULL DEFAULT false,
    confidence numeric(4,3) CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
    source_page integer CHECK (source_page IS NULL OR source_page >= 1),
    source_heading text,
    source_text text NOT NULL CHECK (length(source_text) > 0),
    table_label text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    extracted_at timestamptz NOT NULL DEFAULT now(),
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK ((value IS NULL AND unit IS NULL) OR (value IS NOT NULL AND unit IS NOT NULL))
);

CREATE TABLE IF NOT EXISTS ingestion_sources (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id uuid NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    source_name text NOT NULL,
    source_url text NOT NULL,
    polling_enabled boolean NOT NULL DEFAULT true,
    last_checked_at timestamptz,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (company_id, source_url)
);

CREATE INDEX IF NOT EXISTS idx_documents_company_period
    ON documents (company_id, year, quarter);

CREATE INDEX IF NOT EXISTS idx_documents_status
    ON documents (status);

CREATE INDEX IF NOT EXISTS idx_document_chunks_document
    ON document_chunks (document_id, ordinal);

CREATE INDEX IF NOT EXISTS idx_metrics_company_period
    ON metrics (company_id, year, quarter);

CREATE INDEX IF NOT EXISTS idx_metrics_category_name
    ON metrics (category, metric_name);

CREATE INDEX IF NOT EXISTS idx_metrics_document
    ON metrics (document_id);

CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS trigger AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_companies_updated_at ON companies;
CREATE TRIGGER trg_companies_updated_at
BEFORE UPDATE ON companies
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_documents_updated_at ON documents;
CREATE TRIGGER trg_documents_updated_at
BEFORE UPDATE ON documents
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_ingestion_sources_updated_at ON ingestion_sources;
CREATE TRIGGER trg_ingestion_sources_updated_at
BEFORE UPDATE ON ingestion_sources
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

INSERT INTO companies (code, name, ri_base_url)
VALUES
    ('MRV', 'MRV Engenharia e Participacoes S.A.', 'https://ri.mrv.com.br/informacoes-financeiras/central-de-resultados/'),
    ('CURY', 'Cury Construtora e Incorporadora S.A.', 'https://ri.cury.net/informacoes-aos-investidores/central-de-resultados/'),
    ('TENDA', 'Construtora Tenda S.A.', NULL),
    ('DIRECIONAL', 'Direcional Engenharia S.A.', NULL),
    ('PLANO_PLANO', 'Plano & Plano Desenvolvimento Imobiliario S.A.', NULL),
    ('PACAEMBU', 'Pacaembu Construtora S.A.', NULL)
ON CONFLICT (code) DO UPDATE SET
    ri_base_url = COALESCE(EXCLUDED.ri_base_url, companies.ri_base_url),
    updated_at = now();

INSERT INTO ingestion_sources (company_id, source_name, source_url, polling_enabled)
SELECT id, 'Central de Resultados MRV', 'https://ri.mrv.com.br/informacoes-financeiras/central-de-resultados/', true
FROM companies
WHERE code = 'MRV'
ON CONFLICT (company_id, source_url) DO NOTHING;

INSERT INTO ingestion_sources (company_id, source_name, source_url, polling_enabled)
SELECT id, 'Central de Resultados Cury', 'https://ri.cury.net/informacoes-aos-investidores/central-de-resultados/', true
FROM companies
WHERE code = 'CURY'
ON CONFLICT (company_id, source_url) DO NOTHING;
