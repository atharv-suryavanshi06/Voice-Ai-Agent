import ast
import unittest
from pathlib import Path

from conversation.conversation_manager import ConversationManager


ROOT = Path(__file__).resolve().parents[1]


class ConversationPersistenceTests(unittest.TestCase):
    def test_each_message_is_reserved_once_and_ordered(self):
        manager = ConversationManager(session_id="persist-test")
        manager.process_user_message("hello")
        first = manager.pending_persistence_messages()
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0]["sequence"], 1)

        manager.mark_persistence_queued([first[0]["message_id"]])
        self.assertEqual(manager.pending_persistence_messages(), [])
        manager.mark_persistence_complete([first[0]["message_id"]], True)
        self.assertEqual(manager.pending_persistence_messages(), [])

        manager.record_assistant_reply("hi")
        second = manager.pending_persistence_messages()
        self.assertEqual(len(second), 1)
        self.assertEqual(second[0]["sequence"], 2)
        self.assertNotEqual(first[0]["message_id"], second[0]["message_id"])

    def test_database_disabled_mode_remains_optional(self):
        manager = ConversationManager(session_id="disabled-db")
        self.assertFalse(manager.save_to_db(None))


class LifecycleTests(unittest.TestCase):
    def test_lifecycle_globals_are_initialized_at_module_scope(self):
        tree = ast.parse((ROOT / "main.py").read_text(encoding="utf-8"))
        assignments = {
            node.targets[0].id: node.value.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
        }
        self.assertIsNone(assignments["active_session_manager"])
        self.assertIsNone(assignments["active_db_manager"])
        self.assertIsNone(assignments["active_background_worker"])

    def test_shutdown_no_longer_performs_unconditional_global_save(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertNotIn("if active_session_manager and active_db_manager", source)


if __name__ == "__main__":
    unittest.main()
