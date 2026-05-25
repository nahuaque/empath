from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import anyio

from coach.storage import SurrealStateBackend


class SurrealStateBackendTests(unittest.TestCase):
    def test_memory_backend_round_trips_with_persistent_session(self):
        async def scenario() -> None:
            backend = SurrealStateBackend(
                url="mem://",
                namespace="coach_test",
                database="storage",
                record_id="app_state:memory",
            )
            await backend.save({"version": 1, "users": {"default": {}}})

            self.assertEqual(
                {"version": 1, "users": {"default": {}}},
                await backend.load(),
            )
            await backend.close()

        anyio.run(scenario)

    def test_file_backend_persists_across_backend_instances(self):
        async def scenario() -> None:
            with TemporaryDirectory() as tmpdir:
                url = f"file://{Path(tmpdir) / 'coach-surreal-test.db'}"
                first = SurrealStateBackend(
                    url=url,
                    namespace="coach_test",
                    database="storage",
                    record_id="app_state:file",
                )
                await first.save(
                    {
                        "version": 1,
                        "users": {
                            "default": {
                                "workspace": {"activity_counter": 3},
                            },
                        },
                    }
                )
                await first.close()

                second = SurrealStateBackend(
                    url=url,
                    namespace="coach_test",
                    database="storage",
                    record_id="app_state:file",
                )
                self.assertEqual(
                    {
                        "version": 1,
                        "users": {
                            "default": {
                                "workspace": {"activity_counter": 3},
                            },
                        },
                    },
                    await second.load(),
                )
                await second.close()

        anyio.run(scenario)
