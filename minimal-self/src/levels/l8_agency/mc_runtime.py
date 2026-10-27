"""Bounded game-memory work and coalesced, game-only public state.

Neither a slow embedder nor a livestream reader may block the player's body tick.
"""
import json
import logging
import os
import queue
import threading
import time

logger = logging.getLogger(__name__)


class GameRuntime:
    def __init__(self, snapshot_path=None, capacity=256):
        self.snapshot_path = snapshot_path
        self.queue = queue.Queue(maxsize=capacity)
        self.dropped = 0
        self.errors = 0
        self.memory_errors = 0
        self.snapshot_errors = 0
        self.snapshot_retries = 0
        self.last_error = None
        self._latest = None
        self._lock = threading.Lock()
        self._start_lock = threading.Lock()
        self._stop = threading.Event()
        self._workers = []

    def _start(self):
        with self._start_lock:
            if self._workers or self._stop.is_set():
                return
            self._workers = [threading.Thread(target=self._work, daemon=True, name="mc-memory")]
            if self.snapshot_path:
                self._workers.append(threading.Thread(target=self._publish, daemon=True, name="mc-public-state"))
            for worker in self._workers:
                worker.start()

    def submit(self, fn, *args, **kwargs):
        if self._stop.is_set():
            return False
        self._start()
        try:
            self.queue.put_nowait((fn, args, kwargs))
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def snapshot(self, state):
        if not self.snapshot_path or self._stop.is_set():
            return
        self._start()
        with self._lock:
            self._latest = state

    def _work(self):
        while not self._stop.is_set() or not self.queue.empty():
            try:
                fn, args, kwargs = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                fn(*args, **kwargs)
            except Exception as error:
                self.errors += 1
                self.memory_errors += 1
                self.last_error = "memory:" + type(error).__name__
                logger.warning("[Game] background memory write failed", exc_info=True)
            finally:
                self.queue.task_done()

    def _publish(self):
        while not self._stop.wait(0.1):
            with self._lock:
                state, self._latest = self._latest, None
            if state is None:
                continue
            try:
                os.makedirs(os.path.dirname(os.path.abspath(self.snapshot_path)), exist_ok=True)
                tmp = self.snapshot_path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as stream:
                    json.dump(state, stream, ensure_ascii=False, allow_nan=False)
                for attempt in range(4):
                    try:
                        os.replace(tmp, self.snapshot_path)
                        break
                    except PermissionError:
                        if attempt == 3:
                            raise
                        self.snapshot_retries += 1
                        self._stop.wait(0.01)
            except (OSError, ValueError) as error:
                self.errors += 1
                self.snapshot_errors += 1
                self.last_error = "snapshot:" + type(error).__name__
                if self.snapshot_errors == 1 or self.snapshot_errors % 100 == 0:
                    logger.warning("[Game] public snapshot failed (%s), count=%s", type(error).__name__, self.snapshot_errors)

    def flush(self, timeout=3.0):
        end = time.monotonic() + timeout
        while self.queue.unfinished_tasks and time.monotonic() < end:
            time.sleep(0.01)
        return not self.queue.unfinished_tasks

    def close(self, timeout=3.0):
        self._stop.set()
        deadline = time.monotonic() + timeout
        for worker in self._workers:
            worker.join(max(0, deadline - time.monotonic()))
