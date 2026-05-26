from legal_faiss import LegalFAISS
from intent_router import route
from formatter import format_case_summary, format_irac

rag = LegalFAISS()


def retrieve(query):
    return rag.search(query, top_k=6)


def chat():
    print("⚖️ Legal AI (Clean Architecture v2)")

    while True:
        query = input("\nYou: ").strip()

        if query.lower() == "exit":
            break

        mode = route(query)
        chunks = retrieve(query)

        # =========================
        # CASE SUMMARY MODE
        # =========================
        if mode == "case_summary":
            print(format_case_summary(chunks))

        # =========================
        # SYNTHESIS MODE (TEMP SIMPLE HOOK)
        # =========================
        elif mode == "synthesis":
            print("SYNTHESIS MODE (to be upgraded next)")

        # =========================
        # IRAC MODE
        # =========================
        else:
            print(format_irac(query, chunks))


if __name__ == "__main__":
    chat()