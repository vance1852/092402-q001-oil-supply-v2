from __future__ import annotations

import json
import sqlite3
import unittest
from datetime import datetime, timezone
from decimal import Decimal

from oil_supply.api import JsonApplication
from oil_supply.clock import FrozenClock
from oil_supply.errors import Conflict, Forbidden, ValidationFailed
from oil_supply.planning import AllocationRequest, PricePoint, allocate_capacity, latest_streak
from oil_supply.service import SupplyService
from oil_supply.risk import DemandBucket, inventory_coverage, mark_to_market, supply_gap


class PlanningTests(unittest.TestCase):
    def test_latest_down_streak_counts_day_to_day_moves(self) -> None:
        streak = latest_streak([
            PricePoint("2026-09-18", Decimal("108")),
            PricePoint("2026-09-19", Decimal("105")),
            PricePoint("2026-09-20", Decimal("102")),
            PricePoint("2026-09-21", Decimal("98")),
        ])
        self.assertEqual(streak.direction, "down")
        self.assertEqual(streak.sessions, 3)
        self.assertEqual(streak.start_date, "2026-09-18")
        self.assertEqual(streak.end_date, "2026-09-21")
        self.assertEqual(streak.start_close, Decimal("108"))
        self.assertEqual(streak.end_close, Decimal("98"))
        self.assertEqual(streak.change_close, Decimal("-10"))
        self.assertEqual(streak.percent_change, Decimal("-9.2593"))

    def test_seven_closes_in_news_are_six_leg_down_streak(self) -> None:
        streak = latest_streak([
            PricePoint(f"2026-09-{day:02d}", Decimal(close))
            for day, close in enumerate(("110", "108", "105", "102", "100", "98", "96"), start=18)
        ])
        self.assertEqual(streak.direction, "down")
        self.assertEqual(streak.sessions, 6)
        self.assertEqual(streak.start_date, "2026-09-18")
        self.assertEqual(streak.end_date, "2026-09-24")
        self.assertEqual(streak.start_close, Decimal("110"))
        self.assertEqual(streak.end_close, Decimal("96"))
        self.assertEqual(streak.change_close, Decimal("-14"))
        self.assertEqual(streak.percent_change, Decimal("-12.7273"))

    def test_first_price_point_cannot_define_trend(self) -> None:
        self.assertIsNone(latest_streak([PricePoint("2026-09-18", Decimal("108"))]))
        self.assertIsNone(latest_streak([]))

    def test_flat_last_session_breaks_streak(self) -> None:
        self.assertIsNone(latest_streak([
            PricePoint("2026-09-18", Decimal("108")),
            PricePoint("2026-09-19", Decimal("105")),
            PricePoint("2026-09-20", Decimal("105")),
        ]))

    def test_flat_session_in_middle_restarts_the_streak(self) -> None:
        streak = latest_streak([
            PricePoint("2026-09-18", Decimal("110")),
            PricePoint("2026-09-19", Decimal("108")),
            PricePoint("2026-09-20", Decimal("108")),
            PricePoint("2026-09-21", Decimal("106")),
            PricePoint("2026-09-22", Decimal("104")),
        ])
        self.assertEqual(streak.direction, "down")
        self.assertEqual(streak.sessions, 2)
        self.assertEqual(streak.start_date, "2026-09-20")
        self.assertEqual(streak.change_close, Decimal("-4"))

    def test_opposite_direction_restarts_the_streak(self) -> None:
        streak = latest_streak([
            PricePoint("2026-09-18", Decimal("100")),
            PricePoint("2026-09-19", Decimal("98")),
            PricePoint("2026-09-20", Decimal("99")),
            PricePoint("2026-09-21", Decimal("101")),
        ])
        self.assertEqual(streak.direction, "up")
        self.assertEqual(streak.sessions, 2)
        self.assertEqual(streak.start_date, "2026-09-19")
        self.assertEqual(streak.start_close, Decimal("98"))

    def test_same_day_revision_uses_latest_value_and_recomputes(self) -> None:
        # 旧版本 108 会让 18→19→20 形成两次下跌；修订为 112 后，
        # 19 日相对 18 日上涨，连跌只能从 19 日的修订收盘价起算。
        streak = latest_streak([
            PricePoint("2026-09-18", Decimal("110")),
            PricePoint("2026-09-19", Decimal("108")),
            PricePoint("2026-09-19", Decimal("112")),
            PricePoint("2026-09-20", Decimal("107")),
        ])
        self.assertEqual(streak.sessions, 1)
        self.assertEqual(streak.start_date, "2026-09-19")
        self.assertEqual(streak.start_close, Decimal("112"))
        self.assertEqual(streak.change_close, Decimal("-5"))

    def test_unsorted_input_is_ordered_by_trade_date(self) -> None:
        streak = latest_streak([
            PricePoint("2026-09-20", Decimal("102")),
            PricePoint("2026-09-18", Decimal("108")),
            PricePoint("2026-09-19", Decimal("105")),
        ])
        self.assertEqual(streak.sessions, 2)
        self.assertEqual(streak.start_date, "2026-09-18")

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

    def test_seven_closes_summary_reports_six_move_streak(self) -> None:
        for day, close in zip(range(18, 25), ("110", "108", "105", "102", "100", "98", "96")):
            self.quote(day, close)
        summary = self.service.price_summary("BRENT")
        streak = summary["latest_streak"]
        self.assertEqual(streak["direction"], "down")
        self.assertEqual(streak["sessions"], 6)
        self.assertEqual(streak["start_date"], "2026-09-18")
        self.assertEqual(streak["end_date"], "2026-09-24")
        self.assertEqual(streak["start_close"], "110")
        self.assertEqual(streak["end_close"], "96")
        self.assertEqual(streak["change_close"], "-14")
        self.assertEqual(summary["window"], {
            "requested_sessions": 20,
            "observations": 7,
            "start_date": "2026-09-18",
            "end_date": "2026-09-24",
            "truncated": False,
        })

    def test_summary_window_is_truncated_to_recent_sessions(self) -> None:
        for day, close in zip(range(18, 25), ("100", "99", "98", "97", "96", "95", "94")):
            self.quote(day, close)
        summary = self.service.price_summary("BRENT", 3)
        self.assertEqual(summary["observations"], 3)
        self.assertTrue(summary["window"]["truncated"])
        self.assertEqual(summary["window"]["start_date"], "2026-09-22")
        self.assertEqual(summary["window"]["end_date"], "2026-09-24")
        # 截断后只看到三个收盘点：两次日间下跌，基准点是窗口内最早的收盘价。
        self.assertEqual(summary["latest_streak"]["sessions"], 2)
        self.assertEqual(summary["latest_streak"]["start_date"], "2026-09-22")
        self.assertEqual(summary["latest_streak"]["start_close"], "96")

    def test_summary_with_single_close_has_no_streak(self) -> None:
        self.quote(23, "98")
        summary = self.service.price_summary("BRENT")
        self.assertIsNone(summary["latest_streak"])
        self.assertEqual(summary["latest"]["trade_date"], "2026-09-23")
        self.assertEqual(summary["observations"], 1)

    def test_summary_flat_session_breaks_the_streak(self) -> None:
        for day, close in ((22, "100"), (23, "98"), (24, "98")):
            self.quote(day, close)
        self.assertIsNone(self.service.price_summary("BRENT")["latest_streak"])

    def test_summary_recomputes_after_same_day_revision(self) -> None:
        for day, close in ((22, "100"), (23, "102"), (24, "101")):
            self.quote(day, close)
        # 23 日旧值 102 时：22→23 上涨、23→24 下跌，只有一次下跌。
        streak = self.service.price_summary("BRENT")["latest_streak"]
        self.assertEqual(streak["sessions"], 1)
        self.assertEqual(streak["start_date"], "2026-09-23")
        # 修订 23 日为 104 后仍然是一次下跌，但基准收盘价随之更新。
        self.service.record_quote("plan", {"price_index": "BRENT", "trade_date": "2026-09-23", "close_usd": "104", "source_revision": "r-23-corrected", "observed_at": "2026-09-23T22:00:00Z"})
        revised = self.service.price_summary("BRENT")["latest_streak"]
        self.assertEqual(revised["sessions"], 1)
        self.assertEqual(revised["start_close"], "104")
        self.assertEqual(revised["change_close"], "-3")

    def test_summary_rejects_non_positive_sessions(self) -> None:
        self.quote(23, "98")
        with self.assertRaises(ValidationFailed):
            self.service.price_summary("BRENT", 0)

    def test_api_summary_matches_service_summary(self) -> None:
        for day, close in zip(range(18, 25), ("110", "108", "105", "102", "100", "98", "96")):
            self.quote(day, close)
        app = JsonApplication(self.service)
        response = app.handle("GET", "/quotes/summary/BRENT?sessions=20", {"X-Actor-Id": "risk"})
        self.assertEqual(response.status, 200)
        self.assertEqual(response.body, self.service.price_summary("BRENT", 20))

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
