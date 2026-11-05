"""Non-blocking single-key reads for Windows, macOS and Linux terminals."""

from __future__ import annotations

import sys


class KeyReader:
    def __init__(self) -> None:
        self._old = None
        self.enabled = sys.stdin.isatty()

    def __enter__(self) -> "KeyReader":
        if self.enabled and sys.platform != "win32":
            import termios
            import tty

            fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        return self

    def __exit__(self, *exc) -> None:
        if self._old is not None:
            import termios

            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old)

    def read(self) -> str | None:
        if not self.enabled:
            return None
        if sys.platform == "win32":
            import msvcrt

            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ("\x00", "\xe0"):  # arrow/function key prefix
                    msvcrt.getwch()
                    return None
                return ch.lower()
            return None
        import select

        r, _, _ = select.select([sys.stdin], [], [], 0)
        if r:
            return sys.stdin.read(1).lower()
        return None
