import re
import tempfile
import unittest
from pathlib import Path

from ingestion.markdown_reindex import (
    build_candidate_collection,
    create_candidate_plan,
    validate_written_candidate,
)
from ingestion.recipe import build_ingestion_identity
from ingestion.source_locator import (
    MarkdownSourceValidationError,
    resolve_markdown_policy_sources,
)
from rag.chunker import SemanticChunker
from rag.models import Chunk
from rag.vector_store import PolicyVectorStore


class FAQChunkerTests(unittest.TestCase):
    def test_non_faq_chunk_text_and_ids_keep_legacy_behavior(self):
        text = "Heading\n\nFirst sentence. Second.\n\nTail"
        chunks = SemanticChunker(chunk_size=30, chunk_overlap=5).split_text_to_chunks(
            text, "POL-1", "Policy One"
        )
        self.assertEqual(
            [(chunk.chunk_id, chunk.chunk_text) for chunk in chunks],
            [
                ("POL-1_chunk_0", "Heading\n\n"),
                ("POL-1_chunk_1", "First sentence. Second.\n\nTail"),
            ],
        )

    def test_markdown_and_plain_faq_pairs_are_standalone(self):
        text = (
            "# Frequently Asked Questions\n\n"
            "**Q1. What is covered?**\nThe complete first answer.\n\n"
            "Q2. Is renewal allowed?\nThe complete second answer.\n\n"
            "---\n\n## Definitions\nA definition follows."
        )
        chunks = SemanticChunker(chunk_size=120, chunk_overlap=10).split_text_to_chunks(
            text, "POL-1", "Policy One"
        )
        first = next(chunk.chunk_text for chunk in chunks if "Q1." in chunk.chunk_text)
        second = next(chunk.chunk_text for chunk in chunks if "Q2." in chunk.chunk_text)
        self.assertIn("The complete first answer.", first)
        self.assertIn("The complete second answer.", second)
        self.assertNotIn("Q2.", first)
        self.assertNotIn("Definitions", second)

    def test_long_faq_splits_only_answer_and_repeats_question(self):
        question = "Q1. Explain the long benefit?\n"
        answer = " ".join(f"answer{i}" for i in range(40)) + "\n\n"
        text = question + answer + "Definitions\nA separate definition."
        chunks = SemanticChunker(chunk_size=100, chunk_overlap=10).split_text_to_chunks(
            text, "POL-1", "Policy One"
        )
        faq_chunks = [chunk.chunk_text for chunk in chunks if "Q1." in chunk.chunk_text]
        self.assertGreater(len(faq_chunks), 1)
        self.assertTrue(all(chunk.startswith(question) for chunk in faq_chunks))
        self.assertTrue(all(len(chunk) <= 100 for chunk in faq_chunks))
        self.assertNotIn("Definitions", "".join(faq_chunks))
        for index in range(40):
            self.assertIn(f"answer{index}", "".join(faq_chunks))

    def test_inline_and_malformed_markers_are_not_treated_as_faqs(self):
        text = "This line mentions Q1. inline.\n\nQX. malformed marker\nOrdinary text."
        units = SemanticChunker(chunk_size=200, chunk_overlap=10)._split_into_units(text)
        self.assertFalse(any(type(unit).__name__ == "_FAQUnit" for unit in units))

    def test_all_204_repository_faq_pairs_are_complete_in_one_chunk(self):
        data_dir = Path(__file__).resolve().parents[1] / "Data"
        question_pattern = re.compile(
            r"(?im)^[ \t]*(?:\*\*)?Q[ \t]*\d+[ \t]*[.)][^\r\n]*(?:\r?\n|$)"
        )
        next_section_pattern = re.compile(
            r"(?m)^(?:[ \t]*---+[ \t]*\r?\n(?:[ \t]*\r?\n)*)?(?=#{1,6}[ \t]+\S)"
        )
        pair_count = 0
        for source_path in sorted(data_dir.glob("*.md")):
            text = source_path.read_text(encoding="utf-8")
            questions = list(question_pattern.finditer(text))
            chunks = SemanticChunker().split_text_to_chunks(
                text, source_path.stem, source_path.stem
            )
            for index, question in enumerate(questions):
                end = questions[index + 1].start() if index + 1 < len(questions) else len(text)
                if index + 1 == len(questions):
                    boundary = next_section_pattern.search(text, question.end(), end)
                    if boundary:
                        end = boundary.start()
                pair = text[question.start():end]
                self.assertLessEqual(
                    len(pair),
                    500,
                    f"FAQ exceeds current one-chunk contract in {source_path.name}",
                )
                self.assertTrue(
                    any(pair in chunk.chunk_text for chunk in chunks),
                    f"FAQ pair was split in {source_path.name}: {question.group(0).strip()}",
                )
                pair_count += 1
        self.assertEqual(pair_count, 204)


class MarkdownSourceLocatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def catalog(policy_id="POL/2026/001"):
        return [{"policy_id": policy_id, "policy_name": "Policy One", "policy_code": "P-ONE"}]

    def write(self, name, text):
        path = self.data_dir / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_repeated_embedded_id_is_allowed_and_filename_is_irrelevant(self):
        self.write(
            "unrelated-name.md",
            "| **Product Code** | P-ONE |\n"
            "| **Policy Number** | POL/2026/001 |\n"
            "**Policy Number:** POL/2026/001\n",
        )
        sources = resolve_markdown_policy_sources(self.data_dir, self.catalog())
        self.assertEqual(list(sources), ["POL/2026/001"])
        self.assertEqual(sources["POL/2026/001"].path.name, "unrelated-name.md")
        self.assertEqual(sources["POL/2026/001"].policy_code, "P-ONE")

    def test_unknown_policy_aborts_and_reports_missing_active_policy(self):
        self.write("unknown.md", "**Policy Number:** UNKNOWN/2026/9\n")
        with self.assertRaises(MarkdownSourceValidationError) as raised:
            resolve_markdown_policy_sources(self.data_dir, self.catalog())
        self.assertIn("unknown policy number", str(raised.exception))
        self.assertIn("missing Markdown source", str(raised.exception))

    def test_duplicate_sources_abort(self):
        body = "**Product Code:** P-ONE\n**Policy Number:** POL/2026/001\n"
        self.write("one.md", body)
        self.write("two.md", body)
        with self.assertRaises(MarkdownSourceValidationError) as raised:
            resolve_markdown_policy_sources(self.data_dir, self.catalog())
        self.assertIn("duplicate Markdown sources", str(raised.exception))

    def test_multi_id_document_aborts(self):
        self.write(
            "mixed.md",
            "**Policy Number:** POL/2026/001\n**Policy Number:** POL/2026/002\n",
        )
        catalog = self.catalog() + [{"policy_id": "POL/2026/002", "policy_name": "Policy Two"}]
        with self.assertRaises(MarkdownSourceValidationError) as raised:
            resolve_markdown_policy_sources(self.data_dir, catalog)
        self.assertIn("multiple policy numbers", str(raised.exception))


class IngestionRecipeTests(unittest.TestCase):
    def test_recipe_identity_is_stable_and_changes_with_material_settings(self):
        first = SemanticChunker(500, 50)
        same = SemanticChunker(500, 50)
        base_id, base_recipe = build_ingestion_identity(
            source_hash="a" * 64,
            source_format="markdown",
            chunker=first,
            embedding_model="embed-v1",
        )
        same_id, same_recipe = build_ingestion_identity(
            source_hash="a" * 64,
            source_format="markdown",
            chunker=same,
            embedding_model="embed-v1",
        )
        changed_id, _ = build_ingestion_identity(
            source_hash="a" * 64,
            source_format="markdown",
            chunker=SemanticChunker(450, 50),
            embedding_model="embed-v1",
        )
        model_id, _ = build_ingestion_identity(
            source_hash="a" * 64,
            source_format="markdown",
            chunker=first,
            embedding_model="embed-v2",
        )
        self.assertEqual((base_id, base_recipe), (same_id, same_recipe))
        self.assertNotEqual(base_id, changed_id)
        self.assertNotEqual(base_id, model_id)


class FakeCollection:
    def __init__(self):
        self.calls = []

    def upsert(self, **kwargs):
        self.calls.append(kwargs)


class VectorStoreProvenanceTests(unittest.TestCase):
    def test_insert_keeps_provenance_and_protects_identity_fields(self):
        store = object.__new__(PolicyVectorStore)
        store.collection = FakeCollection()
        chunk = Chunk("chunk-1", "POL-1", "Policy One", 0, "text", [0.1, 0.2])
        store.insert_chunks(
            [chunk],
            provenance={
                "source_filename": "one.md",
                "source_hash": "abc",
                "source_format": "markdown",
                "chunking_version": "v1",
                "chunk_size": 500,
                "chunk_overlap": 50,
                "embedding_model": "embed-v1",
                "policy_code": "P-ONE",
                "policy_id": "WRONG",
            },
            id_suffix="recipe",
        )
        call = store.collection.calls[0]
        self.assertEqual(call["ids"], ["chunk-1__recipe"])
        self.assertEqual(call["metadatas"][0]["policy_id"], "POL-1")
        self.assertEqual(call["metadatas"][0]["policy_code"], "P-ONE")
        self.assertEqual(call["metadatas"][0]["embedding_model"], "embed-v1")


class CandidateBuildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp.name)
        (self.data_dir / "source.md").write_text(
            "**Product Code:** P-ONE\n**Policy Number:** POL/2026/001\n\n"
            "# FAQ\n\nQ1. What is covered?\nEverything stated here.",
            encoding="utf-8",
        )
        self.catalog = [{"policy_id": "POL/2026/001", "policy_name": "Policy One", "policy_code": "P-ONE"}]

    def tearDown(self):
        self.temp.cleanup()

    def plan(self):
        return create_candidate_plan(
            data_dir=self.data_dir,
            catalog_entries=self.catalog,
            collection_name="candidate_v1",
            active_collection_name="insurance_policies",
            embedding_model="embed-v1",
        )

    def test_candidate_build_writes_only_new_named_collection_with_provenance(self):
        created = []

        class FakeStore:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.inserts = []
                self.rows = []
                created.append(self)

                class Collection:
                    def __init__(inner_self, owner):
                        inner_self.owner = owner

                    def get(inner_self, include):
                        return {
                            "ids": [row[0] for row in inner_self.owner.rows],
                            "metadatas": [row[1] for row in inner_self.owner.rows],
                        }

                self.collection = Collection(self)

            def insert_chunks(self, chunks, **kwargs):
                self.inserts.append((chunks, kwargs))
                for chunk in chunks:
                    metadata = dict(kwargs["provenance"])
                    metadata.update({
                        "policy_id": chunk.policy_id,
                        "policy_name": chunk.policy_name,
                        "chunk_index": chunk.chunk_index,
                        "ingestion_status": "active",
                    })
                    self.rows.append((f"{chunk.chunk_id}__{kwargs['id_suffix']}", metadata))

        def embed(chunks, model_name):
            self.assertEqual(model_name, "embed-v1")
            for chunk in chunks:
                chunk.embedding = [0.1, 0.2]
            return chunks

        result = build_candidate_collection(
            self.plan(),
            db_path=self.data_dir / "chroma",
            embedding_generator=embed,
            vector_store_factory=FakeStore,
            collection_exists=lambda _path, _name: False,
        )
        self.assertEqual(created[0].kwargs["collection_name"], "candidate_v1")
        self.assertTrue(created[0].kwargs["create_only"])
        provenance = created[0].inserts[0][1]["provenance"]
        self.assertEqual(provenance["source_format"], "markdown")
        self.assertEqual(provenance["policy_code"], "P-ONE")
        self.assertFalse(result["activated"])
        self.assertTrue(result["verified"])

    def test_structural_validator_rejects_recipe_mismatch(self):
        plan = self.plan()
        rows = []
        for policy in plan.policies:
            for chunk in policy.chunks:
                metadata = dict(policy.provenance)
                metadata.update({
                    "policy_id": chunk.policy_id,
                    "policy_name": chunk.policy_name,
                    "chunk_index": chunk.chunk_index,
                    "ingestion_status": "active",
                })
                rows.append((f"{chunk.chunk_id}__{policy.recipe_hash[:16]}", metadata))
        rows[0][1]["ingestion_recipe_hash"] = "wrong"

        class Collection:
            def get(self, include):
                return {
                    "ids": [row[0] for row in rows],
                    "metadatas": [row[1] for row in rows],
                }

        with self.assertRaisesRegex(RuntimeError, "ingestion_recipe_hash mismatch"):
            validate_written_candidate(plan, Collection())

    def test_existing_candidate_aborts_before_embedding_or_store_creation(self):
        calls = []

        def embed(*_args, **_kwargs):
            calls.append("embed")

        def factory(**_kwargs):
            calls.append("store")

        with self.assertRaises(FileExistsError):
            build_candidate_collection(
                self.plan(),
                embedding_generator=embed,
                vector_store_factory=factory,
                collection_exists=lambda _path, _name: True,
            )
        self.assertEqual(calls, [])

    def test_failed_write_removes_only_the_new_incomplete_candidate(self):
        deleted = []

        class Client:
            def delete_collection(self, name):
                deleted.append(name)

        class FailingStore:
            def __init__(self, **_kwargs):
                self.client = Client()

            def insert_chunks(self, _chunks, **_kwargs):
                raise RuntimeError("simulated candidate write failure")

        def embed(chunks, model_name):
            self.assertEqual(model_name, "embed-v1")
            for chunk in chunks:
                chunk.embedding = [0.1, 0.2]
            return chunks

        with self.assertRaisesRegex(RuntimeError, "simulated candidate write failure"):
            build_candidate_collection(
                self.plan(),
                embedding_generator=embed,
                vector_store_factory=FailingStore,
                collection_exists=lambda _path, _name: False,
            )
        self.assertEqual(deleted, ["candidate_v1"])

    def test_active_collection_name_is_rejected(self):
        with self.assertRaises(ValueError):
            create_candidate_plan(
                data_dir=self.data_dir,
                catalog_entries=self.catalog,
                collection_name="insurance_policies",
                active_collection_name="insurance_policies",
            )


if __name__ == "__main__":
    unittest.main()
