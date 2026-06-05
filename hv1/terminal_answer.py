"""Terminal tool for manually answering cached Teacher questions."""
from __future__ import annotations

from cache_store import CACHE_PATH, set_answer, unanswered_questions


def show_options(options: dict[str, str]) -> None:
    for label in sorted(options):
        print(f"  {label}. {options[label]}")


def main() -> None:
    print(f"Reading cache: {CACHE_PATH}")
    while True:
        questions = unanswered_questions()
        if not questions:
            print("No unanswered questions.")
            return

        for index, item in enumerate(questions, start=1):
            print("\n" + "=" * 72)
            print(f"{index}/{len(questions)} | id={item['id']} | seen={item.get('seen_count', 1)}")
            print(item["question"])
            options = item.get("options") or {}
            if isinstance(options, dict) and options:
                show_options(options)

            answer = input("Answer A/B/C/D, Enter=skip, q=quit: ").strip().upper()
            if answer == "Q":
                return
            if not answer:
                continue
            if answer[:1] not in "ABCD":
                print("Invalid answer, use A/B/C/D.")
                continue
            set_answer(item["id"], answer[:1])
            print(f"Saved {answer[:1]} for {item['id']}")


if __name__ == "__main__":
    main()
