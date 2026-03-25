"""车流量统计模块：通过虚拟计数线判断自行车穿越事件 (增强防抖版)"""
from __future__ import annotations

from campus_bike_detection.models import CountLine, FlowEvent, Track


class FlowCounter:
    def __init__(self, count_lines: list[CountLine]) -> None:
        self._count_lines = count_lines
        self._events: list[FlowEvent] = []
        # 改为记录上次跨线时的 track.age (绝对存活帧数)，用于设置严格的冷却期
        self._last_cross_age: dict[tuple[int, str], int] = {}

    @staticmethod
    def _segments_intersect(
        p1: tuple[float, float],
        p2: tuple[float, float],
        p3: tuple[float, float],
        p4: tuple[float, float],
    ) -> bool:
        """判断线段 p1→p2 与 p3→p4 是否相交（叉积跨立实验）"""
        def cross(o, a, b) -> float:
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)

        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False

    @staticmethod
    def _crossing_direction(
        line: CountLine,
        prev_pos: tuple[float, float],
        curr_pos: tuple[float, float],
    ) -> str:
        """用运动方向与计数线法向量的点积判断穿越方向"""
        dx = line.end[0] - line.start[0]
        dy = line.end[1] - line.start[1]
        nx, ny = -dy, dx  # 法向量
        mx = curr_pos[0] - prev_pos[0]
        my = curr_pos[1] - prev_pos[1]
        dot = nx * mx + ny * my
        return "forward" if dot > 0 else "backward"

    def update(self, tracks: list[Track], timestamp: float) -> list[FlowEvent]:
        """检测轨迹是否穿越计数线，返回本帧产生的穿越事件列表"""
        frame_events: list[FlowEvent] = []

        for track in tracks:
            traj = track.trajectory
            if len(traj) < 2:
                continue

            prev_pos = traj[-2]
            curr_pos = traj[-1]

            for line in self._count_lines:
                key = (track.track_id, line.line_id)

                # =========================================================
                # 核心防抖逻辑：获取该 ID 上次跨线时的绝对生命值 (age)
                # =========================================================
                last_cross_age = self._last_cross_age.get(key, -1)
                
                # 设置 30 帧 (约 1 秒) 的严格冷却期。
                # 只要一辆车跨过线，接下来 1 秒内无论它怎么倒退、框怎么抖动，都绝对不会再算一次。
                if last_cross_age >= 0 and (track.age - last_cross_age) < 30:
                    continue

                if not self._segments_intersect(prev_pos, curr_pos, line.start, line.end):
                    continue

                direction = self._crossing_direction(line, prev_pos, curr_pos)

                # 方向过滤
                if line.direction != "both" and direction != line.direction:
                    continue

                event = FlowEvent(
                    track_id=track.track_id,
                    line_id=line.line_id,
                    timestamp=timestamp,
                    direction=direction,
                )
                
                # 记录这辆车跨线时的准确生命值，开启冷却锁
                self._last_cross_age[key] = track.age
                self._events.append(event)
                frame_events.append(event)

        return frame_events

    def get_flow_stats(self, window_seconds: float = 60.0) -> dict[str, int]:
        """返回各计数线在最近 window_seconds 内的穿越事件数量"""
        result = {line.line_id: 0 for line in self._count_lines}
        if not self._events:
            return result
        latest_ts = self._events[-1].timestamp
        cutoff = latest_ts - window_seconds
        for event in self._events:
            if event.timestamp >= cutoff and event.line_id in result:
                result[event.line_id] += 1
        return result
