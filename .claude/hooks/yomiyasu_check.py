"""PostToolUse フック: docs/ の下の Markdown と AGENTS.md を書き換えたら、yomiyasu の確認を返す。

標準入力にフックの JSON（tool_input.file_path）を受け取る。対象のファイルなら次を確かめ、
見つかったものを additionalContext として Claude に返す。何もなければ何も出さない。

- yomiyasu のリンター（スキル同梱の scripts/yomiyasu_lint.py）の警告
- 60字を超える文、読点が3つ以上ある文（表・見出し・コード・数式の行は除く）
- 会話の中でだけ意味を持つ否定の形（「外した」「やめた」「とは呼ばない」「ではなく」など）
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

LINTER = Path.home() / ".claude" / "skills" / "yomiyasu" / "scripts" / "yomiyasu_lint.py"
# 「帯を外した時刻」の「外した」は別の意味なので、文末か理由が続く形だけを拾う。
NEGATION = re.compile(r"(外した(?:。|ので|ため|$)|やめた|やめる|とは呼ばない|とは言わない|ではなく|使わなかった|入れなかった|しなかった)")
MAX_ITEMS = 25


def target(path: Path, root: Path) -> bool:
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    if rel.as_posix() == "AGENTS.md":
        return True
    return rel.parts[:1] == ("docs",) and path.suffix.lower() == ".md"


def sentences(text: str):
    in_code = False
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        stripped = line.strip()
        if in_code or not stripped or stripped.startswith(("#", "|", "\\[", "\\]", "$$", ">")):
            continue
        body = re.sub(r"^\s*(?:[-*] |\d+\. )", "", line)
        for s in re.split(r"(?<=。)", body):
            s = s.strip()
            if s:
                yield lineno, s


def main() -> int:
    try:
        data = json.loads(sys.stdin.buffer.read().decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return 0
    tool_input = data.get("tool_input") or {}
    raw = tool_input.get("file_path") or (data.get("tool_response") or {}).get("filePath")
    if not raw:
        return 0
    root = Path(os.environ.get("CLAUDE_PROJECT_DIR") or Path(__file__).resolve().parents[2])
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    if not path.is_file() or not target(path, root):
        return 0

    text = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    notes = []
    if LINTER.is_file():
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        out = subprocess.run([sys.executable, str(LINTER), str(path)], capture_output=True, env=env,
                             timeout=60).stdout.decode("utf-8", errors="replace")
        notes += [ln.strip() for ln in out.splitlines() if "[WARN]" in ln or "[ERROR]" in ln]

    issues = []
    for lineno, s in sentences(text):
        commas = s.count("、")
        if len(s) > 60 or commas >= 3:
            issues.append(f"L{lineno} [{len(s)}字・読点{commas}] {s[:80]}")
        if NEGATION.search(s):
            issues.append(f"L{lineno} [否定・経緯の形] {s[:80]}")

    if not notes and not issues:
        return 0
    lines = [f"yomiyasu の確認（{path.name}）。直すか、規則の例外として理由を言えるものだけ残すこと。"]
    lines += notes
    lines += issues[:MAX_ITEMS]
    if len(issues) > MAX_ITEMS:
        lines.append(f"ほか {len(issues) - MAX_ITEMS} 件")
    # ensure_ascii で ASCII だけにして出す。Windows の標準出力の文字コードに左右されない。
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PostToolUse",
                                             "additionalContext": "\n".join(lines)}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
