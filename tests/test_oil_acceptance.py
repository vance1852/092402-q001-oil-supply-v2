from __future__ import annotations

import unittest
from pathlib import Path

from oil_supply.acceptance import run


ROOT = Path(__file__).resolve().parents[1]


class OilSupplyAcceptanceTests(unittest.TestCase):
    def test_offline_acceptance_reports_six_move_down_streak(self) -> None:
        result = run(ROOT)
        self.assertEqual(result["status"], "ok")
        price = result["price"]
        self.assertEqual(price["observations"], 7)
        streak = price["latest_streak"]
        self.assertEqual(streak["direction"], "down")
        # 七个逐日走低的收盘点 = 六次日间下跌，而不是七连跌。
        self.assertEqual(streak["sessions"], 6)
        self.assertEqual(streak["start_date"], "2026-09-18")
        self.assertEqual(streak["end_date"], "2026-09-24")
        self.assertEqual(streak["start_close"], "110")
        self.assertEqual(streak["end_close"], "96")
        self.assertEqual(streak["change_close"], "-14")
        self.assertEqual(price["window"]["start_date"], "2026-09-18")
        self.assertEqual(price["window"]["end_date"], "2026-09-24")
        self.assertFalse(price["window"]["truncated"])
        self.assertTrue(result["audit"]["valid"])


if __name__ == "__main__":
    unittest.main()
