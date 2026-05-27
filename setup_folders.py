"""
setup_folders.py — Run once to create the full folder structure.

Usage:
    python setup_folders.py
"""
import os

FOLDERS = [
    "data/cases/criminal",
    "data/cases/tort",
    "data/cases/contract",
    "data/cases/equity",
    "data/cases/admin",
    "data/cases/family",
    "data/statutes",
    "data/notes",
    "data/profile",
    "data/profile/samples",
    "storage",
    "data",
]

PLACEHOLDER_README = {
    "data/cases/criminal": (
        "Place Singapore criminal law judgments here (.pdf or .txt).\n"
        "Naming convention: CaseName [Year] Court.pdf\n"
        "Example: PP v Low Kok Heng [2007] SGCA 12.pdf\n"
    ),
    "data/cases/tort": "Place tort law judgments here (.pdf or .txt).\n",
    "data/cases/contract": "Place contract law judgments here (.pdf or .txt).\n",
    "data/cases/equity": "Place equity and trusts judgments here (.pdf or .txt).\n",
    "data/cases/admin": "Place administrative law judgments here (.pdf or .txt).\n",
    "data/cases/family": "Place family law judgments here (.pdf or .txt).\n",
    "data/statutes": (
        "Place statute text files here (.pdf or .txt).\n"
        "Download from https://sso.agc.gov.sg\n"
    ),
    "data/notes": "Place research notes, skeleton arguments, or case notes here.\n",
    "data/profile/samples": (
        "Place your past written work here (.pdf or .txt) for the style learner.\n"
        "Run: python style_learner.py bootstrap\n"
    ),
}

def setup():
    print("\n" + "=" * 50)
    print("  Legal AI — Folder Setup")
    print("=" * 50 + "\n")

    created = 0
    existed = 0

    for folder in FOLDERS:
        if os.path.exists(folder):
            print(f"  [OK]      {folder}")
            existed += 1
        else:
            os.makedirs(folder, exist_ok=True)
            print(f"  [CREATED] {folder}")
            created += 1

    for folder, content in PLACEHOLDER_README.items():
        readme_path = os.path.join(folder, "README.txt")
        if not os.path.exists(readme_path):
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(content)

    print(f"\n  Done. {created} folder(s) created, {existed} already existed.")
    print("\n  Next steps:")
    print("  1. Drop case PDFs into the relevant data/cases/<area>/ folders")
    print("  2. Run: python ingest.py")
    print("  3. Run: python app.py")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    setup()