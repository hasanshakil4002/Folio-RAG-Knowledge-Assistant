"""Streaming output cleaner.

Turns the raw LLM token stream into tidy plain text while it streams: removes markdown
marks (**, __, ##, ---, +, backticks) and literal "\\n" sequences. A word that may still be
incomplete (for example a "**" split across two tokens) is held in a buffer until the next
token arrives, so token boundaries never leak formatting characters.
"""
import re

BULLET = "\u2022"

_BR = re.compile(r"<br\s*/?>", re.I)

# line-start rules
_RULE = re.compile(r"^\s*[-_*=~]{3,}\s*$")
_HEAD = re.compile(r"^\s{0,3}#{1,6}\s*")
_QUOTE = re.compile(r"^\s*>+\s?")
_BULLET = re.compile(r"^\s*[-*+\u2022\u25cf\u25aa\u25e6\u2023]\s+")
_NUM = re.compile(r"^\s*(\d+)[.)]\s+")

# inline rules
_STARS = re.compile(r"\*+")
_UNDERS = re.compile(r"(?<!\S)_+|_+(?!\S)|_{2,}")
_TICKS = re.compile(r"`+|~~")
_DASHES = re.compile(r"-{2,}")
_PLUS = re.compile(r"(?:(?<=\s)|^)\+[ \t]")
_CITE = re.compile(r"[ \t]?\[\d{1,3}\]")


class StreamCleaner:
    def __init__(self) -> None:
        self.buf = ""
        self.line_start = True
        self.nl = 0          # consecutive newlines already emitted
        self.started = False
        self.sp = False      # last emitted character was a space

    def feed(self, text: str) -> str:
        self.buf += text
        self.buf = (self.buf.replace("\\r\\n", "\n").replace("\\n", "\n")
                    .replace("\\t", " "))
        self.buf = _BR.sub("\n", self.buf)
        cut = max(self.buf.rfind(" "), self.buf.rfind("\n"), self.buf.rfind("\t"))
        if cut < 0:
            return ""
        head, self.buf = self.buf[: cut + 1], self.buf[cut + 1:]
        return self._clean(head)

    def flush(self) -> str:
        head, self.buf = self.buf, ""
        return self._clean(head).rstrip()

    # ---------- internals ----------
    def _clean(self, head: str) -> str:
        out = []
        for piece in head.splitlines(keepends=True):
            ends_nl = piece.endswith("\n")
            if self.line_start:
                piece = self._line_rules(piece)
            piece = self._inline(piece)
            out.append(self._squeeze(piece))
            self.line_start = ends_nl
        return "".join(out)

    @staticmethod
    def _line_rules(s: str) -> str:
        if _RULE.match(s):
            return ""
        s = _HEAD.sub("", s)
        s = _QUOTE.sub("", s)
        s = _BULLET.sub(BULLET + " ", s)
        s = _NUM.sub(lambda m: f"{m.group(1)}. ", s)
        return s

    @staticmethod
    def _inline(s: str) -> str:
        s = _STARS.sub("", s)
        s = _UNDERS.sub("", s)
        s = _TICKS.sub("", s)
        s = _DASHES.sub("\u2013", s)
        s = _PLUS.sub("", s)
        s = _CITE.sub("", s)
        return s.replace("|", " ")

    def _squeeze(self, s: str) -> str:
        res = []
        for ch in s:
            if ch == "\n":
                if not self.started:
                    continue
                self.nl += 1
                self.sp = False
                if self.nl > 2:
                    continue
                res.append(ch)
            elif ch in " \t":
                if self.nl > 0 or not self.started or self.sp:
                    continue
                self.sp = True
                res.append(ch)
            else:
                self.started = True
                self.nl = 0
                self.sp = False
                res.append(ch)
        return "".join(res)
