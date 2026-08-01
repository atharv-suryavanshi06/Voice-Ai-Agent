import unittest
import json
import tempfile
from email import message_from_string
from email.header import decode_header, make_header
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import database.db_manager as db_module
from database.db_manager import PostgresDBManager
from services.email_sender import EmailService


class RecordingCursor:
    def __init__(self, row=None):
        self.row = row
        self.executions = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, query, params=None):
        self.executions.append((query, params))

    def fetchone(self):
        return self.row


class RecordingConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def cursor(self):
        return self._cursor

    def close(self):
        self.closed = True


class PolicyDocumentDatabaseTests(unittest.TestCase):
    def manager_with_cursor(self, cursor):
        manager = PostgresDBManager.__new__(PostgresDBManager)
        manager.enabled = True
        connection = RecordingConnection(cursor)
        manager._get_connection = lambda: connection
        return manager, connection

    def test_get_policy_document_uses_only_exact_policy_id(self):
        row = (
            "SLI/FHS/2026/00792144",
            "SecureLife Family Health Suraksha",
            "SecureLife Insurance Pvt. Ltd.",
            "Family Floater",
            33512,
            1500000,
            "correct second document",
            None,
            None,
        )
        cursor = RecordingCursor(row)
        manager, connection = self.manager_with_cursor(cursor)

        result = manager.get_policy_document("SLI/FHS/2026/00792144")

        self.assertEqual(result["policy_id"], "SLI/FHS/2026/00792144")
        query, params = cursor.executions[0]
        self.assertIn("WHERE policy_id = %s", query)
        self.assertNotIn("policy_name", query.split("WHERE", 1)[1])
        self.assertNotIn("LIKE", query.upper())
        self.assertEqual(params, ("SLI/FHS/2026/00792144",))
        self.assertTrue(connection.closed)

    def test_exact_source_replacement_backs_up_then_can_clear_pdf(self):
        cursor = RecordingCursor()
        manager, _connection = self.manager_with_cursor(cursor)

        fake_driver = SimpleNamespace(Binary=lambda value: value)
        with mock.patch.object(db_module, "psycopg2", fake_driver):
            saved = manager.save_policy_document(
                policy_id="SLI/FHS/2026/00792144",
                policy_name="SecureLife Family Health Suraksha",
                document_text="correct second document",
                insurer="SecureLife Insurance Pvt. Ltd.",
                plan_type="Family Floater",
                premium=33512,
                sum_insured=1500000,
                pdf_path=None,
                pdf_bytes=None,
                replace_pdf=True,
                backup_id="exact-id-markdown-sync-v1:test",
                backup_reason="test repair",
            )

        self.assertTrue(saved)
        self.assertEqual(len(cursor.executions), 2)
        backup_query, backup_params = cursor.executions[0]
        upsert_query, upsert_params = cursor.executions[1]
        self.assertIn("INSERT INTO policy_ingestions", backup_query)
        self.assertIn("ON CONFLICT (ingestion_id) DO NOTHING", backup_query)
        self.assertEqual(backup_params[-1], "SLI/FHS/2026/00792144")
        self.assertIn("WHEN %s THEN EXCLUDED.pdf_data", upsert_query)
        self.assertIsNone(upsert_params[-2])
        self.assertIs(upsert_params[-1], True)


class ExactIdMarkdownSyncTests(unittest.TestCase):
    FIRST_ID = "SLFH/2026/0518291"
    SECOND_ID = "SLI/FHS/2026/00792144"

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data_dir = self.root / "Data"
        self.data_dir.mkdir()
        self.first_md = self.data_dir / "SecureLife_Family_Policy.md"
        self.second_md = self.data_dir / "SecureLife_Family_Policy (1).md"
        self.first_text = f"**Policy Number:** {self.FIRST_ID}\nFirst terms"
        self.second_text = (
            "| **Policy Number** | " + self.SECOND_ID + " |\nSecond exact terms"
        )
        self.first_md.write_bytes(self.first_text.encode("utf-8"))
        self.second_md.write_bytes(self.second_text.encode("utf-8"))
        self.first_pdf = self.first_md.with_suffix(".pdf")
        self.first_pdf.write_bytes(b"first verified pdf")
        self.catalog = [
            {
                "policy_id": self.FIRST_ID,
                "policy_name": "SecureLife Family Health Suraksha",
                "insurer": "SecureLife Insurance",
                "plan_type": "Family Floater",
                "premium": 18540,
                "sum_insured": 1000000,
            },
            {
                "policy_id": self.SECOND_ID,
                "policy_name": "SecureLife Family Health Suraksha",
                "insurer": "SecureLife Insurance Pvt. Ltd.",
                "plan_type": "Family Floater",
                "premium": 33512,
                "sum_insured": 1500000,
            },
        ]
        self.catalog_path = self.root / "policy_catalog.json"
        self.catalog_path.write_text(json.dumps(self.catalog), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def make_manager(self):
        manager = PostgresDBManager.__new__(PostgresDBManager)
        manager.enabled = True
        records = {
            policy_id: {
                **entry,
                "document_text": "wrong shared document",
                "pdf_path": "wrong-shared.pdf",
                "pdf_data": b"wrong shared pdf",
            }
            for policy_id, entry in (
                (self.FIRST_ID, self.catalog[0]),
                (self.SECOND_ID, self.catalog[1]),
            )
        }
        saves = []

        manager._fetch_policy_document_exact = lambda policy_id: (
            dict(records[policy_id]) if policy_id in records else None
        )
        manager._get_verified_ingestion_pdf = lambda _policy_id: None

        def save_policy_document(**kwargs):
            saves.append(kwargs)
            records[kwargs["policy_id"]] = {
                "policy_id": kwargs["policy_id"],
                "policy_name": kwargs["policy_name"],
                "insurer": kwargs["insurer"],
                "plan_type": kwargs["plan_type"],
                "premium": kwargs["premium"],
                "sum_insured": kwargs["sum_insured"],
                "document_text": kwargs["document_text"],
                "pdf_path": kwargs["pdf_path"],
                "pdf_data": kwargs["pdf_bytes"],
            }
            return True

        manager.save_policy_document = save_policy_document
        return manager, records, saves

    def test_duplicate_names_sync_by_embedded_id_and_clear_unverified_pdf(self):
        manager, records, saves = self.make_manager()

        synced = manager.sync_existing_markdown_documents(
            data_dir=str(self.data_dir),
            catalog_path=str(self.catalog_path),
        )

        self.assertTrue(synced)
        self.assertTrue(manager.is_policy_document_verified(self.FIRST_ID))
        self.assertTrue(manager.is_policy_document_verified(self.SECOND_ID))
        self.assertEqual(len(saves), 2)
        by_id = {item["policy_id"]: item for item in saves}
        self.assertEqual(by_id[self.FIRST_ID]["document_text"], self.first_text)
        self.assertEqual(by_id[self.FIRST_ID]["pdf_bytes"], b"first verified pdf")
        self.assertEqual(Path(by_id[self.FIRST_ID]["pdf_path"]), self.first_pdf.resolve())
        self.assertEqual(by_id[self.SECOND_ID]["document_text"], self.second_text)
        self.assertIsNone(by_id[self.SECOND_ID]["pdf_path"])
        self.assertIsNone(by_id[self.SECOND_ID]["pdf_bytes"])
        self.assertTrue(by_id[self.SECOND_ID]["replace_pdf"])
        self.assertTrue(by_id[self.SECOND_ID]["backup_id"])
        self.assertIsNone(records[self.SECOND_ID]["pdf_data"])

        # A retry is a real no-op: no new update or backup attempt.
        manager.sync_existing_markdown_documents(
            data_dir=str(self.data_dir),
            catalog_path=str(self.catalog_path),
        )
        self.assertEqual(len(saves), 2)

    def test_invalid_source_set_aborts_before_database_reads_or_writes(self):
        manager, _records, saves = self.make_manager()
        reads = []
        manager._fetch_policy_document_exact = lambda policy_id: reads.append(policy_id)
        (self.data_dir / "Unknown.md").write_text(
            "**Policy Number:** UNKNOWN/2026/123\nUnknown terms",
            encoding="utf-8",
        )

        synced = manager.sync_existing_markdown_documents(
            data_dir=str(self.data_dir),
            catalog_path=str(self.catalog_path),
        )

        self.assertFalse(synced)
        self.assertFalse(manager.is_policy_document_verified(self.FIRST_ID))
        self.assertFalse(manager.is_policy_document_verified(self.SECOND_ID))
        self.assertEqual(reads, [])
        self.assertEqual(saves, [])

    def test_active_exact_id_ingestion_is_valid_attachment_provenance(self):
        manager, _records, saves = self.make_manager()
        recorded_path = self.root / "recorded-second-policy.pdf"

        def verified_pdf(policy_id):
            if policy_id == self.SECOND_ID:
                return {
                    "pdf_path": str(recorded_path),
                    "pdf_data": b"recorded exact-id pdf",
                }
            return None

        manager._get_verified_ingestion_pdf = verified_pdf
        manager.sync_existing_markdown_documents(
            data_dir=str(self.data_dir),
            catalog_path=str(self.catalog_path),
        )

        by_id = {item["policy_id"]: item for item in saves}
        self.assertEqual(by_id[self.SECOND_ID]["pdf_bytes"], b"recorded exact-id pdf")
        self.assertEqual(
            Path(by_id[self.SECOND_ID]["pdf_path"]),
            recorded_path.resolve(),
        )


class FakeSMTP:
    sent_message = None

    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def starttls(self):
        return None

    def login(self, *_args):
        return None

    def sendmail(self, _sender, _recipients, message):
        type(self).sent_message = message


class FakeDocumentDB:
    def __init__(self, verified=True):
        self.lookups = []
        self.logs = []
        self.verified = verified

    def get_policy_document(self, policy_id):
        self.lookups.append(policy_id)
        return {
            "policy_id": policy_id,
            "document_text": "Exact policy terms",
            "pdf_path": None,
            "pdf_data": b"%PDF-exact-policy",
        }

    def log_sent_email(self, **kwargs):
        self.logs.append(kwargs)
        return True

    def is_policy_document_verified(self, _policy_id):
        return self.verified


def duplicate_policy(policy_id, policy_code, premium):
    return SimpleNamespace(
        policy_id=policy_id,
        policy_code=policy_code,
        policy_name="SecureLife Family Health Suraksha",
        insurer="SecureLife Insurance",
        plan_type="Family Floater",
        premium=premium,
        sum_insured=1000000,
        covers_diabetes=True,
        covers_hypertension=True,
    )


class DuplicatePolicyEmailTests(unittest.TestCase):
    def setUp(self):
        FakeSMTP.sent_message = None
        self.first = duplicate_policy("SLFH/2026/0518291", "SLFH-2026", 18540)
        self.second = duplicate_policy(
            "SLI/FHS/2026/00792144",
            "SLIHLIP26002V010001",
            33512,
        )
        self.service = EmailService(policy_catalog=[self.first, self.second])
        self.service.sender_email = "sender@example.com"
        self.service.app_password = "test-password"

    def test_duplicate_names_use_code_number_and_id_safe_attachment(self):
        database = FakeDocumentDB()

        with mock.patch("services.email_sender.smtplib.SMTP", FakeSMTP):
            sent = self.service.send_policy_recommendation_email(
                recipient_email="customer@example.com",
                customer_name="customer",
                policies=[self.first, self.second],
                db_manager=database,
                session_id="session-1",
            )

        self.assertTrue(sent)
        self.assertEqual(database.lookups, ["SLFH/2026/0518291"])
        self.assertEqual(database.logs[0]["policy_id"], "SLFH/2026/0518291")
        self.assertEqual(
            database.logs[0]["policy_name"],
            "SecureLife Family Health Suraksha",
        )

        message = message_from_string(FakeSMTP.sent_message)
        subject = str(make_header(decode_header(message["Subject"])))
        self.assertIn("code SLFH-2026", subject)
        self.assertIn("policy number SLFH/2026/0518291", subject)

        html_part = next(
            part for part in message.walk() if part.get_content_type() == "text/html"
        )
        html = html_part.get_payload(decode=True).decode("utf-8")
        self.assertIn("code SLFH-2026", html)
        self.assertIn("policy number SLI/FHS/2026/00792144", html)

        attachments = [
            part for part in message.walk()
            if part.get_content_disposition() == "attachment"
        ]
        self.assertEqual(len(attachments), 1)
        self.assertEqual(
            attachments[0].get_filename(),
            "SecureLife_Family_Health_Suraksha_SLFH_2026_0518291_Policy_Document.pdf",
        )

    def test_single_recommended_duplicate_still_uses_catalog_disambiguation(self):
        database = FakeDocumentDB()
        with mock.patch("services.email_sender.smtplib.SMTP", FakeSMTP):
            sent = self.service.send_policy_recommendation_email(
                recipient_email="customer@example.com",
                customer_name=None,
                policies=[self.second],
                db_manager=database,
            )

        self.assertTrue(sent)
        message = message_from_string(FakeSMTP.sent_message)
        subject = str(make_header(decode_header(message["Subject"])))
        self.assertIn("code SLIHLIP26002V010001", subject)
        self.assertIn("policy number SLI/FHS/2026/00792144", subject)
        attachment = next(
            part for part in message.walk()
            if part.get_content_disposition() == "attachment"
        )
        self.assertIn("SLI_FHS_2026_00792144", attachment.get_filename())

    def test_failed_exact_id_sync_suppresses_polluted_document_and_attachment(self):
        database = FakeDocumentDB(verified=False)
        with mock.patch("services.email_sender.smtplib.SMTP", FakeSMTP):
            sent = self.service.send_policy_recommendation_email(
                recipient_email="customer@example.com",
                customer_name="customer",
                policies=[self.second],
                db_manager=database,
            )

        self.assertTrue(sent)
        message = message_from_string(FakeSMTP.sent_message)
        html_part = next(
            part for part in message.walk() if part.get_content_type() == "text/html"
        )
        html = html_part.get_payload(decode=True).decode("utf-8")
        self.assertNotIn("Exact policy terms", html)
        attachments = [
            part for part in message.walk()
            if part.get_content_disposition() == "attachment"
        ]
        self.assertEqual(attachments, [])


if __name__ == "__main__":
    unittest.main()
