"""
Smoke tests for everything that doesn't need the Brev VM.

Tests:
  1. Slack connectivity — can we reach the API with the bot token?
  2. Post to channel — does chat.postMessage work?
  3. Join notification — does the "Julia is listening now" message post?
  4. YES/NO confirmation card — do the buttons render correctly?
  5. Huddle detector — does the AppleScript run without crashing?

Run:
  python test_slack.py
"""
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

import httpx

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
SLACK_CHANNEL   = os.environ.get("SLACK_CHANNEL", "")


def check(label: str, ok: bool, detail: str = "") -> None:
    icon = "✅" if ok else "❌"
    print(f"  {icon}  {label}" + (f" — {detail}" if detail else ""))


# ------------------------------------------------------------------
# 1. Token present
# ------------------------------------------------------------------
async def test_tokens() -> bool:
    print("\n[1] Checking env vars")
    bot_ok = bool(SLACK_BOT_TOKEN)
    ch_ok  = bool(SLACK_CHANNEL)
    check("SLACK_BOT_TOKEN set", bot_ok)
    check("SLACK_CHANNEL set",   ch_ok, SLACK_CHANNEL if ch_ok else "missing")
    return bot_ok and ch_ok


# ------------------------------------------------------------------
# 2. Slack API reachable
# ------------------------------------------------------------------
async def test_api_reachable() -> bool:
    print("\n[2] Slack API connectivity")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                "https://slack.com/api/auth.test",
                headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
            )
            data = resp.json()
            ok   = data.get("ok", False)
            check("auth.test", ok, data.get("bot_id", data.get("error", "")))
            return ok
    except Exception as exc:
        check("auth.test", False, str(exc))
        return False


# ------------------------------------------------------------------
# 3. Post a plain message
# ------------------------------------------------------------------
async def test_post_message() -> bool:
    print("\n[3] Post plain message to channel")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                "https://slack.com/api/chat.postMessage",
                headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
                json={"channel": SLACK_CHANNEL, "text": "🧪 Julia smoke test — plain message"},
            )
            data = resp.json()
            ok   = data.get("ok", False)
            check("chat.postMessage", ok, data.get("ts", data.get("error", "")))
            return ok
    except Exception as exc:
        check("chat.postMessage", False, str(exc))
        return False


# ------------------------------------------------------------------
# 4. Join notification
# ------------------------------------------------------------------
async def test_join_notification() -> bool:
    print("\n[4] Join notification")
    from meeting_summary import post_join_notification
    try:
        await post_join_notification(SLACK_CHANNEL)
        check("post_join_notification", True)
        return True
    except Exception as exc:
        check("post_join_notification", False, str(exc))
        return False


# ------------------------------------------------------------------
# 5. YES/NO confirmation card
# ------------------------------------------------------------------
async def test_confirmation_card() -> bool:
    print("\n[5] YES/NO confirmation card")
    from schema import TaskProposal
    from confirmation import _post_card

    proposal = TaskProposal(
        recipients=[],
        content="🧪 Test: drinks at 7:30 tonight",
        voice_prompt="Should I post a message about drinks tonight at 7:30?",
    )
    try:
        ts = await _post_card(proposal)
        check("Block Kit card posted", bool(ts), f"ts={ts}")
        return bool(ts)
    except Exception as exc:
        check("Block Kit card posted", False, str(exc))
        return False


# ------------------------------------------------------------------
# 6. Huddle detector (just checks AppleScript runs)
# ------------------------------------------------------------------
async def test_huddle_detector() -> bool:
    print("\n[6] Huddle detector (AppleScript)")
    from huddle_detector import _huddle_active
    try:
        result = await _huddle_active()
        check("AppleScript ran without error", True, f"huddle_active={result}")
        return True
    except Exception as exc:
        check("AppleScript ran without error", False, str(exc))
        return False


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------
async def main() -> None:
    print("=" * 50)
    print("  Julia — Slack smoke tests")
    print("=" * 50)

    results = []
    results.append(await test_tokens())
    if not results[-1]:
        print("\n⛔ Missing env vars — fix .env and re-run.")
        return

    results.append(await test_api_reachable())
    results.append(await test_post_message())
    results.append(await test_join_notification())
    results.append(await test_confirmation_card())
    results.append(await test_huddle_detector())

    passed = sum(results)
    total  = len(results)
    print(f"\n{'=' * 50}")
    print(f"  {passed}/{total} tests passed")
    if passed == total:
        print("  All good — ready to test with Brev VM next.")
    else:
        print("  Fix the failures above before merging.")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
