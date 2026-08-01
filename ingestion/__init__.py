"""
ingestion package

Exposes the document ingestion pipeline.
"""

from ingestion.pdf_processor import process_policy_pdf
from ingestion.models import PolicyMetadata

__all__ = ["process_policy_pdf", "PolicyMetadata"]
