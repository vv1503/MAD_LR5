"""Реализация алгоритма CLOPE для кластеризации категориальных транзакций."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Hashable, Iterable, List, Sequence


Item = Hashable
Transaction = Sequence[Item]


@dataclass
class Cluster:
    """Кластер: частоты элементов, площадь S, ширина W, число транзакций N."""

    occ: Counter = field(default_factory=Counter)
    S: int = 0
    W: int = 0
    N: int = 0

    def add(self, transaction: Transaction) -> None:
        """Добавляет транзакцию в кластер и обновляет гистограмму."""
        for item in transaction:
            if self.occ[item] == 0:
                self.W += 1
            self.occ[item] += 1
            self.S += 1
        self.N += 1

    def remove(self, transaction: Transaction) -> None:
        """Удаляет транзакцию из кластера и обновляет гистограмму."""
        for item in transaction:
            self.occ[item] -= 1
            self.S -= 1
            if self.occ[item] == 0:
                del self.occ[item]
                self.W -= 1
        self.N -= 1

    def delta_add(self, transaction: Transaction, r: float) -> float:
        """Приращение Profit при добавлении транзакции (DeltaAdd)."""
        if self.N == 0:
            s_new = len(transaction)
            w_new = len(set(transaction))
            return (s_new * 1.0) / (w_new ** r)

        s_new = self.S + len(transaction)
        w_new = self.W
        for item in set(transaction):
            if self.occ[item] == 0:
                w_new += 1
        new_term = s_new * (self.N + 1) / (w_new ** r)
        old_term = self.S * self.N / (self.W ** r) if self.W > 0 else 0.0
        return new_term - old_term

    def height(self) -> float:
        """Средняя высота гистограммы H = S / W."""
        return self.S / self.W if self.W else 0.0

    def gradient(self, r: float) -> float:
        """Градиент кластера G = S / W^r."""
        return self.S / (self.W ** r) if self.W else 0.0


class CLOPE:
    """Алгоритм CLOPE: максимизация Profit(C, r) для транзакционных данных."""

    def __init__(self, repulsion: float = 2.0, max_iter: int = 20, allow_new: bool = True):
        """Инициализация: repulsion r, max_iter, разрешение новых кластеров."""
        self.repulsion = repulsion
        self.max_iter = max_iter
        self.allow_new = allow_new
        self.clusters_: List[Cluster] = []
        self.labels_: List[int] = []
        self.profit_history_: List[float] = []

    def _profit(self) -> float:
        """Глобальный критерий Profit(C, r)."""
        total_n = sum(c.N for c in self.clusters_)
        if total_n == 0:
            return 0.0
        num = sum(c.gradient(self.repulsion) * c.N for c in self.clusters_ if c.N > 0)
        return num / total_n

    def _best_cluster(self, transaction: Transaction, current: int | None = None) -> int:
        """Индекс кластера с максимальным DeltaAdd (или -1 для нового)."""
        best_idx = -1
        best_delta = float("-inf")

        for idx, cluster in enumerate(self.clusters_):
            if cluster.N == 0 and current is None:
                continue
            delta = cluster.delta_add(transaction, self.repulsion)
            if current is not None and idx == current:
                # транзакция уже в кластере; сравниваем с альтернативами
                continue
            if delta > best_delta:
                best_delta = delta
                best_idx = idx

        if self.allow_new:
            empty = Cluster()
            new_delta = empty.delta_add(transaction, self.repulsion)
            if new_delta > best_delta:
                return -1
        return best_idx

    def fit(self, transactions: Iterable[Transaction]) -> "CLOPE":
        """Обучение: фаза инициализации и итеративное уточнение."""
        data = [list(t) for t in transactions]
        n = len(data)
        self.clusters_ = []
        self.labels_ = [-1] * n
        self.profit_history_ = []

        # Фаза 1: инициализация
        for i, t in enumerate(data):
            idx = self._best_cluster(t)
            if idx == -1:
                self.clusters_.append(Cluster())
                idx = len(self.clusters_) - 1
            self.clusters_[idx].add(t)
            self.labels_[i] = idx
        self.profit_history_.append(self._profit())

        # Фаза 2: итерации
        for _ in range(self.max_iter):
            moved = False
            for i, t in enumerate(data):
                cur = self.labels_[i]
                self.clusters_[cur].remove(t)

                idx = self._best_cluster(t)
                if idx == -1:
                    self.clusters_.append(Cluster())
                    idx = len(self.clusters_) - 1

                if idx != cur:
                    moved = True

                self.clusters_[idx].add(t)
                self.labels_[i] = idx

            # переиндексация без пустых кластеров
            mapping = {}
            new_clusters = []
            for old_idx, cluster in enumerate(self.clusters_):
                if cluster.N > 0:
                    mapping[old_idx] = len(new_clusters)
                    new_clusters.append(cluster)
            self.clusters_ = new_clusters
            self.labels_ = [mapping[lab] for lab in self.labels_]
            self.profit_history_.append(self._profit())

            if not moved:
                break

        return self

    def fit_predict(self, transactions: Iterable[Transaction]) -> List[int]:
        """Обучение и возврат меток кластеров."""
        self.fit(transactions)
        return list(self.labels_)
