import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ingestion.models import PolicyMetadata
from ingestion.pdf_processor import process_policy_pdf


class FakeChunker:
    def split_text_to_chunks(self, text, policy_id, policy_name):
        from rag.models import Chunk
        return [Chunk(f"{policy_id}_chunk_0", policy_id, policy_name, 0, text)]


class FakeVectorStore:
    def __init__(self, fail_stage=False, fail_activate=False):
        self.fail_stage = fail_stage
        self.fail_activate = fail_activate
        self.staged = set()
        self.active = set()

    def stage_policy_chunks(self, policy_id, chunks, ingestion_id):
        if self.fail_stage:
            raise RuntimeError("chroma write failed")
        self.staged.add(ingestion_id)
        return [chunks[0].chunk_id]

    def activate_staged_policy(self, policy_id, ingestion_id, remove_previous=False):
        if self.fail_activate:
            raise RuntimeError("chroma activation failed")
        self.active.add(ingestion_id)

    def remove_previous_policy_versions(self, policy_id, ingestion_id):
        return None

    def delete_ingestion(self, ingestion_id):
        self.staged.discard(ingestion_id)
        self.active.discard(ingestion_id)


class FakeDB:
    def __init__(self, stage_result=True, activate_result=True):
        self.enabled = True
        self.stage_result = stage_result
        self.activate_result = activate_result
        self.failed = []
        self.deleted = []

    def stage_policy_document(self, **_kwargs):
        return self.stage_result

    def get_policy_document(self, _policy_id):
        return None

    def activate_staged_policy_document(self, _ingestion_id):
        return self.activate_result

    def mark_policy_ingestion_failed(self, ingestion_id, error):
        self.failed.append((ingestion_id, error))
        return True

    def delete_policy_document(self, _policy_id):
        self.deleted.append(_policy_id)
        return True

    def save_policy_document(self, **_kwargs):
        return True


def metadata():
    return PolicyMetadata(
        policy_id="POL-1",
        policy_name="Test Policy",
        insurer="Test Insurer",
        plan_type="Individual",
        premium=1000,
        min_age=18,
        max_age=65,
        sum_insured=100000,
        smoker_allowed=True,
        covers_diabetes=True,
        covers_hypertension=True,
        parents_allowed=False,
        children_allowed=False,
    )


class IngestionRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.pdf = root / "policy.pdf"
        self.pdf.write_bytes(b"stable fake pdf")
        self.catalog = root / "catalog.json"
        self.original = [{"policy_id": "OLD", "policy_name": "Old"}]
        self.catalog.write_text(json.dumps(self.original), encoding="utf-8")
        self.journal = root / "manifest.json"

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def embed(chunks):
        for chunk in chunks:
            chunk.embedding = [0.1, 0.2]
        return chunks

    def run_ingestion(self, **overrides):
        options = dict(
            text_extractor=lambda _path: "insurance policy premium coverage",
            metadata_parser=lambda _text: metadata(),
            chunker=FakeChunker(),
            embedding_generator=self.embed,
            vector_store=FakeVectorStore(),
            db_manager=FakeDB(),
            journal_path=str(self.journal),
        )
        options.update(overrides)
        return process_policy_pdf(str(self.pdf), str(self.catalog), **options)

    def assert_original_catalog(self):
        self.assertEqual(json.loads(self.catalog.read_text(encoding="utf-8")), self.original)

    def test_major_stage_failures_do_not_publish_policy(self):
        cases = [
            {"text_extractor": lambda _path: (_ for _ in ()).throw(RuntimeError("extract"))},
            {"metadata_parser": lambda _text: (_ for _ in ()).throw(ValueError("metadata"))},
            {"embedding_generator": lambda _chunks: (_ for _ in ()).throw(RuntimeError("embedding"))},
            {"vector_store": FakeVectorStore(fail_stage=True)},
            {"db_manager": FakeDB(stage_result=False)},
            {"vector_store": FakeVectorStore(fail_activate=True)},
            {"db_manager": FakeDB(activate_result=False)},
        ]
        for overrides in cases:
            with self.subTest(overrides=list(overrides)):
                self.catalog.write_text(json.dumps(self.original), encoding="utf-8")
                with self.assertRaises(RuntimeError):
                    self.run_ingestion(**overrides)
                self.assert_original_catalog()

    def test_catalogue_publication_failure_cleans_staged_vectors(self):
        vector = FakeVectorStore()
        with mock.patch("ingestion.pdf_processor._atomic_write_catalog", side_effect=OSError("publish")):
            with self.assertRaises(RuntimeError):
                self.run_ingestion(vector_store=vector)
        self.assertFalse(vector.active)
        self.assertFalse(vector.staged)
        self.assert_original_catalog()

    def test_final_catalogue_failure_compensates_database_activation(self):
        import ingestion.pdf_processor as processor

        vector = FakeVectorStore()
        database = FakeDB()
        original_write = processor._atomic_write_catalog
        calls = {"count": 0}

        def fail_final(path, catalog):
            calls["count"] += 1
            if calls["count"] == 2:
                raise OSError("final publish")
            return original_write(path, catalog)

        with mock.patch("ingestion.pdf_processor._atomic_write_catalog", side_effect=fail_final):
            with self.assertRaises(RuntimeError):
                self.run_ingestion(vector_store=vector, db_manager=database)

        self.assertEqual(database.deleted, ["POL-1"])
        self.assertFalse(vector.active)
        self.assert_original_catalog()

    def test_retry_is_idempotent_and_activates_one_catalog_entry(self):
        vector = FakeVectorStore()
        database = FakeDB()
        self.run_ingestion(vector_store=vector, db_manager=database)
        self.run_ingestion(vector_store=vector, db_manager=database)
        catalog = json.loads(self.catalog.read_text(encoding="utf-8"))
        self.assertEqual(sum(1 for item in catalog if item["policy_id"] == "POL-1"), 1)
        self.assertEqual(len(vector.active), 1)


if __name__ == "__main__":
    unittest.main()
