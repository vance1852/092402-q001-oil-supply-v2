from __future__ import annotations

import json
import sqlite3
import unittest
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from oil_supply.api import JsonApplication
from oil_supply.clock import FrozenClock
from oil_supply.errors import Conflict, Forbidden
from oil_supply.planning import AllocationRequest, PricePoint, allocate_capacity, latest_streak
from oil_supply.service import SupplyService
from oil_supply.risk import DemandBucket, inventory_coverage, mark_to_market, supply_gap


ROOT = Path(__file__).resolve().parents[1]


class PlanningTests(unittest.TestCase):
    def down_points(self, closes: list[str], start_day: int = 17) -> list[PricePoint]:
        return [
            PricePoint(f"2026-09-{start_day + index:02d}", Decimal(close))
            for index, close in enumerate(closes)
        ]

    def test_six_day_decline_requires_seven_closes(self) -> None:
        # 新闻所述六连跌：七个逐日走低的收盘点，只有六次相邻日间变动。
        streak = latest_streak(self.down_points(["111", "108", "105", "102", "100", "98", "96"]))
        self.assertEqual(streak.direction, "down")
        self.assertEqual(streak.sessions, 6)
        self.assertEqual(streak.start_date, "2026-09-17")
        self.assertEqual(streak.end_date, "2026-09-23")
        self.assertEqual(streak.start_close, Decimal("111"))
        self.assertEqual(streak.end_close, Decimal("96"))
        self.assertEqual(streak.change, Decimal("-15"))
        self.assertTrue(streak.truncated)

    def test_six_lower_closes_are_only_five_moves(self) -> None:
        # 缺陷回归：六个走低收盘点只有五次日间变动，不能报成六连跌；
        # 累计跌幅必须从第一个点（变动前基准价）算起。
        streak = latest_streak(self.down_points(["108", "105", "102", "100", "98", "96"], start_day=18))
        self.assertEqual(streak.direction, "down")
        self.assertEqual(streak.sessions, 5)
        self.assertEqual(streak.start_date, "2026-09-18")
        self.assertEqual(streak.end_date, "2026-09-23")
        self.assertEqual(streak.start_close, Decimal("108"))
        self.assertEqual(streak.end_close, Decimal("96"))
        self.assertEqual(streak.change, Decimal("-12"))
        self.assertEqual(streak.percent_change, Decimal("-11.1111"))

    def test_single_point_cannot_establish_trend(self) -> None:
        self.assertIsNone(latest_streak([PricePoint("2026-09-23", Decimal("96"))]))
        self.assertIsNone(latest_streak([]))

    def test_flat_last_session_breaks_streak(self) -> None:
        # 末日平盘打断此前的下跌：没有截至最新交易日的连涨/连跌。
        streak = latest_streak(self.down_points(["108", "105", "102", "102"], start_day=18))
        self.assertIsNone(streak)

    def test_prior_flat_resets_streak_base(self) -> None:
        # 中段平盘打断：只统计平盘之后的连续变动，起点为平盘收盘点。
        streak = latest_streak(self.down_points(["108", "105", "105", "103", "101"], start_day=18))
        self.assertEqual(streak.direction, "down")
        self.assertEqual(streak.sessions, 2)
        self.assertEqual(streak.start_date, "2026-09-20")
        self.assertEqual(streak.start_close, Decimal("105"))
        self.assertEqual(streak.change, Decimal("-4"))
        self.assertFalse(streak.truncated)

    def test_up_streak_counts_moves_and_changes_direction_break(self) -> None:
        streak = latest_streak(self.down_points(["100", "98", "99", "101", "103"], start_day=18))
        self.assertEqual(streak.direction, "up")
        self.assertEqual(streak.sessions, 3)
        self.assertEqual(streak.start_date, "2026-09-19")
        self.assertEqual(streak.start_close, Decimal("98"))
        self.assertEqual(streak.end_close, Decimal("103"))
        self.assertEqual(streak.change, Decimal("5"))

    def test_same_day_revision_uses_latest_value(self) -> None:
        # 同一交易日给出多个点时以最后一个为准（模拟同日修订）。
        points = self.down_points(["108", "105", "102"], start_day=21)
        points.append(PricePoint("2026-09-23", Decimal("101")))
        streak = latest_streak(points)
        self.assertEqual(streak.sessions, 2)
        self.assertEqual(streak.end_close, Decimal("101"))
        self.assertEqual(streak.change, Decimal("-7"))

    def test_allocation_is_stable_and_does_not_exceed_capacity(self) -> None:
        rows = allocate_capacity(Decimal("100"), [
            AllocationRequest("later", Decimal("80"), 20, "2026-09-24T09:00:00Z"),
            AllocationRequest("first", Decimal("70"), 10, "2026-09-24T10:00:00Z"),
        ])
        self.assertEqual(rows[0]["nomination_id"], "first")
        self.assertEqual(rows[0]["allocated_barrels"], "70.000")
        self.assertEqual(rows[1]["allocated_barrels"], "30.000")

    def test_inventory_coverage_and_supply_gap(self) -> None:
        coverage = inventory_coverage(
            [{"facility_id": "terminal", "product": "gasoline-92", "available_barrels": "250"}],
            [DemandBucket("terminal", "gasoline-92", Decimal("100"), Decimal("20"))],
        )
        self.assertEqual(coverage[0]["coverage_days"], "2.30")
        self.assertTrue(coverage[0]["below_three_days"])
        gap = supply_gap(
            opening_inventory=Decimal("100"),
            confirmed_inbound=Decimal("30"),
            forecast_demand=Decimal("120"),
            protected_reserve=Decimal("40"),
        )
        self.assertEqual(gap["supply_gap"], "30.000")

    def test_mark_to_market_groups_deterministically(self) -> None:
        result = mark_to_market(
            [{"position_id": "p1", "price_index": "BRENT", "quantity_barrels": "100", "entry_price_usd": "105"}],
            {"BRENT": Decimal("98")},
        )
        self.assertEqual(result["unrealized_pnl_usd"], "-700.00")


class SupplyServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.connection = sqlite3.connect(":memory:", isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.clock = FrozenClock(datetime(2026, 9, 24, 8, 0, tzinfo=timezone.utc))
        self.service = SupplyService(self.connection, self.clock)
        for user_id, role in (("plan", "planner"), ("dispatch", "dispatcher"), ("risk", "risk"), ("audit", "auditor")):
            self.service.create_user(user_id, user_id, role)
        self.service.create_facility("plan", {"facility_id": "field-a", "name": "北部油田", "kind": "storage", "timezone": "Asia/Shanghai", "capacity_barrels": "500000"})
        self.service.create_facility("plan", {"facility_id": "terminal-b", "name": "沿海终端", "kind": "terminal", "timezone": "Asia/Shanghai", "capacity_barrels": "800000"})
        self.service.create_route("plan", {"route_id": "pipe-a-b", "origin_id": "field-a", "destination_id": "terminal-b", "product": "crude", "daily_capacity": "100000", "loss_basis_points": 25, "transit_hours": 36})

    def tearDown(self) -> None:
        self.connection.close()

    def quote(self, day: int, close: str) -> dict[str, object]:
        return self.service.record_quote("plan", {"price_index": "BRENT", "trade_date": f"2026-09-{day}", "close_usd": close, "source_revision": f"r-{day}", "observed_at": f"2026-09-{day}T21:00:00Z"})

    def test_quote_revisions_preserve_history(self) -> None:
        first = self.quote(23, "98")
        second = self.service.record_quote("plan", {"price_index": "BRENT", "trade_date": "2026-09-23", "close_usd": "97.8", "source_revision": "r-23-corrected", "observed_at": "2026-09-23T22:00:00Z"})
        self.assertNotEqual(first["quote_id"], second["quote_id"])
        rows = self.connection.execute("SELECT * FROM price_index_quotes ORDER BY quote_id").fetchall()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["supersedes_quote_id"], rows[0]["quote_id"])

    def record_closes(self, closes: list[tuple[int, str]]) -> None:
        for day, close in closes:
            self.quote(day, close)

    def test_price_summary_seven_closes_is_six_day_decline(self) -> None:
        # 新闻所述六连跌需要七个收盘点。
        self.record_closes([
            (17, "111"), (18, "108"), (19, "105"), (20, "102"),
            (21, "100"), (22, "98"), (23, "96"),
        ])
        summary = self.service.price_summary("BRENT")
        streak = summary["latest_streak"]
        self.assertEqual(streak["direction"], "down")
        self.assertEqual(streak["sessions"], 6)
        self.assertEqual(streak["start_date"], "2026-09-17")
        self.assertEqual(streak["end_date"], "2026-09-23")
        self.assertEqual(streak["start_close"], "111")
        self.assertEqual(streak["end_close"], "96")
        self.assertEqual(streak["change"], "-15")
        self.assertEqual(summary["observations"], 7)

    def test_price_summary_six_closes_is_only_five_day_decline(self) -> None:
        # 缺陷回归：六个收盘点只是五连跌。
        self.record_closes([(18, "108"), (19, "105"), (20, "102"), (21, "100"), (22, "98"), (23, "96")])
        streak = self.service.price_summary("BRENT")["latest_streak"]
        self.assertEqual(streak["sessions"], 5)
        self.assertEqual(streak["start_date"], "2026-09-18")
        self.assertEqual(streak["change"], "-12")

    def test_streak_recomputes_after_same_day_revision(self) -> None:
        self.record_closes([(21, "108"), (22, "105"), (23, "102")])
        streak = self.service.price_summary("BRENT")["latest_streak"]
        self.assertEqual(streak["sessions"], 2)
        self.assertEqual(streak["end_close"], "102")
        # 修订最后一个交易日的收盘价为平盘：连跌被打断。
        self.service.record_quote("plan", {"price_index": "BRENT", "trade_date": "2026-09-23", "close_usd": "105", "source_revision": "r-23-corrected", "observed_at": "2026-09-23T22:00:00Z"})
        self.assertIsNone(self.service.price_summary("BRENT")["latest_streak"])
        # 修订为更高的收盘价：按修订后价格重算为单日下跌后转涨的结构。
        self.service.record_quote("plan", {"price_index": "BRENT", "trade_date": "2026-09-23", "close_usd": "106", "source_revision": "r-23-final", "observed_at": "2026-09-23T23:00:00Z"})
        streak = self.service.price_summary("BRENT")["latest_streak"]
        self.assertEqual(streak["direction"], "up")
        self.assertEqual(streak["sessions"], 1)
        self.assertEqual(streak["start_close"], "105")
        self.assertEqual(streak["end_close"], "106")
        self.assertEqual(streak["change"], "1")

    def test_streak_uses_full_history_beyond_query_window(self) -> None:
        # 窗口只截断观测数与均线，不能截断连跌段。
        self.record_closes([
            (14, "120"), (15, "117"), (16, "114"), (17, "111"),
            (18, "108"), (19, "105"), (20, "102"), (21, "100"),
            (22, "98"), (23, "96"),
        ])
        summary = self.service.price_summary("BRENT", sessions=5)
        self.assertEqual(summary["observations"], 5)
        streak = summary["latest_streak"]
        self.assertEqual(streak["sessions"], 9)
        self.assertEqual(streak["start_date"], "2026-09-14")
        self.assertEqual(streak["start_close"], "120")
        self.assertEqual(streak["end_close"], "96")
        # 基准起点不在 5 日窗口内，提示摘要窗口未覆盖整段趋势。
        self.assertTrue(streak["truncated"])
        # 放大窗口到覆盖完整历史后，同一趋势不再标记为截断。
        self.assertFalse(self.service.price_summary("BRENT", sessions=20)["latest_streak"]["truncated"])

    def test_window_truncation_is_flagged(self) -> None:
        # 窗口只保留最近两个收盘点，但连跌的基准起点在窗口之外，标记为截断。
        self.record_closes([(21, "108"), (22, "105"), (23, "102")])
        summary = self.service.price_summary("BRENT", sessions=2)
        self.assertEqual(summary["observations"], 2)
        streak = summary["latest_streak"]
        self.assertEqual(streak["sessions"], 2)
        self.assertEqual(streak["start_date"], "2026-09-21")
        self.assertTrue(streak["truncated"])

    def test_flat_last_session_has_no_streak(self) -> None:
        self.record_closes([(21, "108"), (22, "105"), (23, "105")])
        self.assertIsNone(self.service.price_summary("BRENT")["latest_streak"])

    def test_api_summary_matches_offline_acceptance(self) -> None:
        from oil_supply import acceptance as offline

        result = offline.run(ROOT)
        expected = result["price"]["latest_streak"]
        self.assertEqual(expected["sessions"], 6)
        self.assertEqual(expected["start_date"], "2026-09-17")
        self.assertEqual(expected["end_date"], "2026-09-23")
        self.assertEqual(expected["start_close"], "111")
        self.assertEqual(expected["end_close"], "96")
        self.assertEqual(expected["change"], "-15")
        # 同一报价集通过 HTTP 边界得到完全一致的连涨连跌结论。
        self.record_closes([
            (17, "111"), (18, "108"), (19, "105"), (20, "102"),
            (21, "100"), (22, "98"), (23, "96"),
        ])
        response = JsonApplication(self.service).handle(
            "GET", "/quotes/summary/BRENT", {"X-Actor-Id": "plan"}
        )
        self.assertEqual(response.status, 200)
        self.assertEqual(response.body["latest_streak"], expected)

    def test_nomination_replay_and_payload_conflict(self) -> None:
        payload = {"nomination_id": "nom-1", "route_id": "pipe-a-b", "shipper_id": "refinery", "service_date": "2026-09-25", "requested_barrels": "80000", "priority": 10, "idempotency_key": "key-1"}
        first = self.service.submit_nomination("dispatch", payload)
        self.assertEqual(first, self.service.submit_nomination("dispatch", payload))
        changed = dict(payload, requested_barrels="81000")
        with self.assertRaises(Conflict):
            self.service.submit_nomination("dispatch", changed)

    def test_outage_reduces_allocation_and_transfer_consumes_inventory(self) -> None:
        self.service.announce_outage("risk", "pipe-a-b", "2026-09-25T00:00:00Z", "2026-09-25T23:59:59Z", "50", "检修")
        for number, requested, priority in ((1, "40000", 10), (2, "30000", 20)):
            self.service.submit_nomination("dispatch", {"nomination_id": f"nom-{number}", "route_id": "pipe-a-b", "shipper_id": f"shipper-{number}", "service_date": "2026-09-25", "requested_barrels": requested, "priority": priority, "idempotency_key": f"key-{number}"})
        allocation = self.service.allocate("dispatch", "pipe-a-b", "2026-09-25")
        self.assertEqual(allocation["available_capacity"], "50000.000")
        self.assertEqual(allocation["allocations"][1]["allocated_barrels"], "10000.000")
        self.service.add_inventory_lot("dispatch", {"lot_id": "lot-1", "facility_id": "field-a", "product": "crude", "grade": "BRENT", "quantity_barrels": "60000", "unit_cost_usd": "91", "received_at": "2026-09-24T06:00:00Z"})
        transfer = self.service.dispatch_transfer("dispatch", "transfer-1", "nom-1", "lot-1", 2)
        self.assertEqual(transfer["loaded_barrels"], "40000.000")
        self.assertEqual(self.service.inventory_lot("lot-1")["available_barrels"], "20000.000")

    def test_scenario_is_approved_and_replayed_by_input(self) -> None:
        self.quote(23, "98")
        self.service.add_inventory_lot("dispatch", {"lot_id": "lot-1", "facility_id": "field-a", "product": "crude", "grade": "BRENT", "quantity_barrels": "60000", "unit_cost_usd": "91", "received_at": "2026-09-24T06:00:00Z"})
        self.service.create_scenario("plan", {"scenario_id": "restart", "name": "管道恢复", "price_index_drop_percent": "9", "route_capacity_changes": {"pipe-a-b": "20"}, "demand_changes": {"field-a:crude": "-5"}})
        with self.assertRaises(Forbidden):
            self.service.approve_scenario("plan", "restart", 1)
        self.service.approve_scenario("risk", "restart", 1)
        first = self.service.run_scenario("plan", "restart", "2026-09-23")
        second = self.service.run_scenario("plan", "restart", "2026-09-23")
        self.assertFalse(first["replayed"])
        self.assertTrue(second["replayed"])
        self.assertEqual(first["run_id"], second["run_id"])

    def test_audit_chain_detects_tampering(self) -> None:
        self.assertTrue(self.service.audit_chain("audit")["valid"])
        self.connection.execute("UPDATE supply_audit_events SET payload_json='{}' WHERE event_id=1")
        self.assertFalse(self.service.audit_chain("audit")["valid"])

    def test_api_exposes_browser_free_boundary(self) -> None:
        app = JsonApplication(self.service)
        self.assertEqual(app.handle("GET", "/health").status, 200)
        response = app.handle("GET", "/quotes/summary/BRENT", {"X-Actor-Id": "plan"})
        self.assertEqual(response.status, 404)
        self.assertEqual(response.body["error"]["code"], "not_found")


if __name__ == "__main__":
    unittest.main()
