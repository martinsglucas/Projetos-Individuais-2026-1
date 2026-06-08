from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CompanyCode(StrEnum):
    MRV = "MRV"
    CURY = "CURY"
    TENDA = "TENDA"
    DIRECIONAL = "DIRECIONAL"
    PLANO_PLANO = "PLANO_PLANO"
    PACAEMBU = "PACAEMBU"


class ReportType(StrEnum):
    OPERATIONAL_PREVIEW = "operational_preview"
    EARNINGS_RELEASE = "earnings_release"
    CONJUNTURA_BULLETIN = "conjuntura_bulletin"
    OTHER = "other"


class DocumentStatus(StrEnum):
    DISCOVERED = "discovered"
    DOWNLOADED = "downloaded"
    PARSED = "parsed"
    EXTRACTED = "extracted"
    VALIDATED = "validated"
    FAILED = "failed"
    SKIPPED_DUPLICATE = "skipped_duplicate"


class ExtractionStrategy(StrEnum):
    FULL_SCAN = "full_scan"
    SEMANTIC_CHUNKING = "semantic_chunking"
    HYBRID = "hybrid"


class MetricCategory(StrEnum):
    LANDBANK = "landbank"
    LAUNCHES = "launches"
    SALES = "sales"
    TRANSFERS = "transfers"
    PRODUCTION = "production"
    CASH_GENERATION = "cash_generation"
    VSO = "vso"
    FINANCING = "financing"
    OTHER = "other"


class MetricUnit(StrEnum):
    BRL = "BRL"
    BRL_MILLION = "BRL_million"
    BRL_BILLION = "BRL_billion"
    USD = "USD"
    USD_MILLION = "USD_million"
    UNITS = "units"
    PERCENT = "percent"
    PERCENTAGE_POINTS = "percentage_points"
    THOUSAND_BRL = "thousand_BRL"
    OTHER = "other"


class Period(BaseModel):
    model_config = ConfigDict(extra="forbid")

    year: int = Field(ge=2000, le=2100)
    quarter: int = Field(ge=1, le=4)

    @property
    def label(self) -> str:
        return f"{self.quarter}T{str(self.year)[-2:]}"

    @classmethod
    def from_label(cls, value: str) -> "Period":
        normalized = value.strip().upper().replace(" ", "")
        if len(normalized) != 4 or normalized[1] != "T":
            raise ValueError("period must use format like 1T25")

        quarter = int(normalized[0])
        short_year = int(normalized[2:])
        year = 2000 + short_year
        return cls(year=year, quarter=quarter)


class SourceEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_number: int | None = Field(default=None, ge=1)
    chunk_id: str | None = None
    heading: str | None = None
    raw_text: str = Field(min_length=1)
    table_label: str | None = None


class DocumentMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    company: CompanyCode
    report_type: ReportType = ReportType.OPERATIONAL_PREVIEW
    period: Period
    source_url: str | None = None
    pdf_hash: str = Field(min_length=64, max_length=64)
    storage_path: str | None = None
    original_filename: str | None = None
    discovered_at: datetime | None = None
    downloaded_at: datetime | None = None

    @field_validator("pdf_hash")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        lowered = value.lower()
        if any(char not in "0123456789abcdef" for char in lowered):
            raise ValueError("pdf_hash must be a lowercase or uppercase SHA-256 hex digest")
        return lowered


class DocumentChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    document_hash: str = Field(min_length=64, max_length=64)
    ordinal: int = Field(ge=0)
    heading: str | None = None
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    content: str = Field(min_length=1)
    token_count: int | None = Field(default=None, ge=0)
    parser: str = "docling"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_page_range(self) -> "DocumentChunk":
        if self.page_start and self.page_end and self.page_end < self.page_start:
            raise ValueError("page_end must be greater than or equal to page_start")
        return self


class ExtractedMetric(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    company: CompanyCode
    period: Period
    category: MetricCategory
    metric_name: str = Field(min_length=1)
    segment: str | None = Field(
        default=None,
        description="Business segment such as MRV, Sensia, Resia, Luggo, Urba, or total incorporation.",
    )
    value: Decimal | None = Field(
        default=None,
        description="Absolute value only. Use null when the document does not provide the value.",
    )
    unit: MetricUnit | None = None
    currency: Literal["BRL", "USD"] | None = None
    scale: Literal["unit", "thousand", "million", "billion", "percent", "percentage_points"] | None = None
    is_estimated: bool = False
    confidence: Decimal | None = Field(default=None, ge=Decimal("0"), le=Decimal("1"))
    evidence: SourceEvidence

    @model_validator(mode="after")
    def validate_value_unit_pair(self) -> "ExtractedMetric":
        if self.value is None and self.confidence is not None and self.confidence > Decimal("0.2"):
            raise ValueError("missing values should not have high confidence")
        if self.value is not None and self.unit is None:
            raise ValueError("unit is required when value is present")
        return self


class ExtractionRun(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    document_hash: str = Field(min_length=64, max_length=64)
    strategy: ExtractionStrategy
    parser: str = "docling"
    llm_provider: str
    llm_model: str
    prompt_version: str
    started_at: datetime
    finished_at: datetime | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    status: Literal["running", "succeeded", "failed"] = "running"
    error_message: str | None = None


class OperationalPreviewExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    document: DocumentMetadata
    extraction_run: ExtractionRun
    chunks: list[DocumentChunk] = Field(default_factory=list)
    metrics: list[ExtractedMetric] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class LLMMetricExtractionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    company: CompanyCode
    period: Period
    report_type: ReportType
    metrics: list[ExtractedMetric]
    missing_relevant_fields: list[str] = Field(default_factory=list)
    extraction_notes: list[str] = Field(default_factory=list)
