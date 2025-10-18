# %%
from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import takewhile

import databento as db
from sortedcontainers import SortedDict

from src.logger import configure_logging, get_logger


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


logger = get_logger(__name__)


@dataclass
class PriceLevel:
    price: int
    size: int = 0
    count: int = 0

    def __str__(self) -> str:
        price = self.price / db.FIXED_PRICE_SCALE
        return f"{self.size:4} @ {price:6.2f} | {self.count:2} order(s)"


@dataclass
class LevelOrders:
    price: int
    orders: list[db.MBOMsg] = field(default_factory=list, compare=False)

    def __bool__(self) -> bool:
        return bool(self.orders)

    @property
    def level(self) -> PriceLevel:
        return PriceLevel(
            price=self.price,
            count=sum(1 for o in self.orders if not o.flags & db.RecordFlags.F_TOB),
            size=sum(o.size for o in self.orders),
        )


@dataclass
class Book:
    orders_by_id: dict[int, db.MBOMsg] = field(default_factory=dict)
    offers: SortedDict[int, LevelOrders] = field(default_factory=SortedDict)
    bids: SortedDict[int, LevelOrders] = field(default_factory=SortedDict)

    def bbo(self) -> tuple[PriceLevel | None, PriceLevel | None]:
        return self.get_bid_level(), self.get_ask_level()

    def get_bid_level(self, idx: int = 0) -> PriceLevel | None:
        if self.bids and len(self.bids) > idx:
            # Reverse for bids to get highest prices first
            return self.bids.peekitem(-(idx + 1))[1].level
        return None

    def get_ask_level(self, idx: int = 0) -> PriceLevel | None:
        if self.offers and len(self.offers) > idx:
            return self.offers.peekitem(idx)[1].level
        return None

    def get_bid_level_by_px(self, px: int) -> PriceLevel | None:
        try:
            return self._get_level(px, "B").level
        except KeyError:
            return None

    def get_ask_level_by_px(self, px: int) -> PriceLevel | None:
        try:
            return self._get_level(px, "A").level
        except KeyError:
            return None

    def get_order(self, id: int) -> db.MBOMsg | None:
        return self.orders_by_id.get(id)

    def get_queue_pos(self, id: int) -> int | None:
        order = self.get_order(id)
        if not order:
            return None
        level = self._get_level(order.price, order.side)
        return sum(
            order.size for order in takewhile(lambda o: o.order_id != id, level.orders)
        )

    def get_snapshot(self, level_count: int = 1) -> list[db.BidAskPair]:
        snapshots = []
        for level in range(level_count):
            ba_pair = db.BidAskPair()
            bid = self.get_bid_level(level)
            if bid:
                ba_pair.bid_px = bid.price
                ba_pair.bid_sz = bid.size
                ba_pair.bid_ct = bid.count
            ask = self.get_ask_level(level)
            if ask:
                ba_pair.ask_px = ask.price
                ba_pair.ask_sz = ask.size
                ba_pair.ask_ct = ask.count
            snapshots.append(ba_pair)
        return snapshots

    def apply(self, mbo: db.MBOMsg) -> None:
        # Trade, Fill, or None: no change
        if mbo.action in ("T", "F", "N"):
            return
        # Clear book: remove all resting orders
        if mbo.action == "R":
            self._clear()
            return
        # side=N is only valid with Trade, Fill, and Clear actions
        assert mbo.side in ("A", "B")
        # UNDEF_PRICE indicates the book level should be removed
        if mbo.price == db.UNDEF_PRICE and mbo.flags & db.RecordFlags.F_TOB:
            self._side_levels(mbo.side).clear()
            return
        # Add: insert a new order
        if mbo.action == "A":
            self._add(mbo)
        # Cancel: partially or fully cancel some size from a resting order
        elif mbo.action == "C":
            self._cancel(mbo)
        # Modify: change the price and/or size of a resting order
        elif mbo.action == "M":
            self._modify(mbo)
        else:
            raise ValueError(f"Unknown action={mbo.action}")

    def _clear(self) -> None:
        self.orders_by_id.clear()
        self.offers.clear()
        self.bids.clear()

    def _add(self, mbo: db.MBOMsg) -> None:
        if mbo.flags & db.RecordFlags.F_TOB:
            levels = self._side_levels(mbo.side)
            levels.clear()
            levels[mbo.price] = LevelOrders(price=mbo.price, orders=[mbo])
        else:
            level = self._get_or_insert_level(mbo.price, mbo.side)
            assert mbo.order_id not in self.orders_by_id
            self.orders_by_id[mbo.order_id] = mbo
            level.orders.append(mbo)

    def _cancel(self, mbo: db.MBOMsg) -> None:
        order = self.orders_by_id[mbo.order_id]
        level = self._get_level(mbo.price, mbo.side)
        assert order.size >= mbo.size
        order.size -= mbo.size
        # If the full size is cancelled, remove the order from the book
        if order.size == 0:
            self.orders_by_id.pop(mbo.order_id)
            level.orders.remove(order)
            # If the level is now empty, remove it from the book
            if not level:
                self._remove_level(mbo.price, mbo.side)

    def _modify(self, mbo: db.MBOMsg) -> None:
        order = self.orders_by_id.get(mbo.order_id)
        if order is None:
            # If order not found, treat it as an add
            logger.debug(
                "Order %s not found during modify; treating as new add", mbo.order_id
            )
            self._add(mbo)
            return
        assert order.side == mbo.side, f"Order {order} changed side to {mbo.side}"
        level = self._get_level(order.price, order.side)
        if order.price != mbo.price:
            # Changing price loses priority
            level.orders.remove(order)
            if not level:
                self._remove_level(order.price, mbo.side)
            level = self._get_or_insert_level(mbo.price, mbo.side)
            level.orders.append(mbo)
        elif order.size < mbo.size:
            # Increasing size loses priority
            level.orders.remove(order)
            level.orders.append(mbo)
        else:
            # Update in place
            level.orders[level.orders.index(order)] = mbo
        self.orders_by_id[mbo.order_id] = mbo

    def _side_levels(self, side: str) -> SortedDict:
        if side == "A":
            return self.offers
        if side == "B":
            return self.bids
        raise ValueError(f"Invalid {side =}")

    def _get_level(self, price: int, side: str) -> LevelOrders:
        levels = self._side_levels(side)
        if price not in levels:
            raise KeyError(f"No price level found for {price =} and {side =}")
        return levels[price]

    def _get_or_insert_level(self, price: int, side: str) -> LevelOrders:
        levels = self._side_levels(side)
        if price in levels:
            return levels[price]
        level = LevelOrders(price=price)
        levels[price] = level
        return level

    def _remove_level(self, price: int, side: str) -> None:
        levels = self._side_levels(side)
        levels.pop(price)


@dataclass
class Market:
    books: defaultdict[int, defaultdict[int, Book]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(Book)),
    )

    def get_books_by_pub(self, instrument_id: int) -> defaultdict[int, Book]:
        return self.books[instrument_id]

    def get_book(self, instrument_id: int, publisher_id: int) -> Book:
        return self.books[instrument_id][publisher_id]

    def bbo(
        self,
        instrument_id: int,
        publisher_id: int,
    ) -> tuple[PriceLevel | None, PriceLevel | None]:
        return self.books[instrument_id][publisher_id].bbo()

    def aggregated_bbo(
        self,
        instrument_id: int,
    ) -> tuple[PriceLevel | None, PriceLevel | None]:
        agg_bbo: list[PriceLevel | None] = [None, None]
        # max for bids, min for asks
        for idx, reducer in [(0, max), (1, min)]:
            all_best = [b.bbo()[idx] for b in self.books[instrument_id].values()]
            all_best = [b for b in all_best if b]
            if not all_best:
                continue
            best_price = reducer(b.price for b in all_best)
            best = [b for b in all_best if b.price == best_price]
            agg_bbo[idx] = PriceLevel(
                price=best_price,
                size=sum(b.size for b in best),
                count=sum(b.count for b in best),
            )
        return tuple(agg_bbo)

    def apply(self, mbo: db.MBOMsg) -> None:
        book = self.books[mbo.instrument_id][mbo.publisher_id]
        book.apply(mbo)


def _log_price_level(label: str, level: PriceLevel | None) -> None:
    if level is None:
        logger.info("    %s: None", label)
    else:
        logger.info("    %s: %s", label, level)


def _log_aggregated_book(
    symbol: str, ts: str, bid: PriceLevel | None, ask: PriceLevel | None
) -> None:
    logger.info("%s Aggregated BBO | %s", symbol, ts)
    _log_price_level("Ask", ask)
    _log_price_level("Bid", bid)


# %%
if __name__ == "__main__":
    configure_logging(
        component="order_book",
        debug=_env_flag("ORDER_BOOK_DEBUG"),
        level=os.getenv("ORDER_BOOK_LOG_LEVEL", "INFO"),
        simplified=_env_flag("ORDER_BOOK_SIMPLIFIED_FORMAT", True),
        log_dir=os.getenv("ORDER_BOOK_LOG_DIR"),
        structured_logging=_env_flag("ORDER_BOOK_STRUCTURED_LOGS"),
        include_console=_env_flag("ORDER_BOOK_CONSOLE_OUTPUT", True),
    )

    # First, create a historical client
    client = db.Historical(os.getenv("DATABENTO_API_KEY"))

    # Next, we will request MBO data starting from the beginning of pre-market trading hours
    # or load the file if we've already downloaded it.
    data_path = "./data/raw/databento/20250501/glbx-mdp3-20250501.mbo.dbn"
    if os.path.exists(data_path):
        logger.info("Loading MBO data from %s", data_path)
        data = db.DBNStore.from_file(data_path)
    else:
        logger.info(
            "Fetching MBO data from Databento for symbols %s",
            ["GOOG", "GOOGL"],
        )
        data = client.timeseries.get_range(
            dataset="DBEQ.BASIC",
            start="2024-04-03T08:00:00",
            end="2024-04-03T14:00:00",
            symbols=["GOOG", "GOOGL"],
            schema="mbo",
            path=data_path,
        )

    # Then we parse the symbology into a more usable format
    instrument_map = db.common.symbology.InstrumentMap()
    instrument_map.insert_metadata(data.metadata)

    # Finally, we iterate over each book update
    # %%
    market = Market()
    for mbo in data:
        # And apply it
        market.apply(mbo)
        # If it's the last update in an event, log the state of the aggregated book
        if mbo.flags & db.RecordFlags.F_LAST:
            symbol = (
                instrument_map.resolve(mbo.instrument_id, mbo.pretty_ts_recv.date())
                or ""
            )
            best_bid, best_offer = market.aggregated_bbo(mbo.instrument_id)
            _log_aggregated_book(
                symbol,
                str(mbo.pretty_ts_recv),
                best_bid,
                best_offer,
            )
