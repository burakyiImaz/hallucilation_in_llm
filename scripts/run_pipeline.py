"""Small runnable example that boots the default pipeline and runs sample prompts."""
from pipeline.bootstrap import build_default_runner


def main():
    runner = build_default_runner()

    prompts = [
        "Who was Ada Lovelace and what is she known for?",
        "List three facts about the Moon.",
    ]

    for p in prompts:
        print("\n--- PROMPT ---")
        print(p)
        res = runner.run(p)
        print("Responses:")
        for r in res["responses"]:
            print(" - ", r)
        print("Evaluation:", res["evaluation"]) 
        print("Decision:", res["decision"]) 


if __name__ == "__main__":
    main()
