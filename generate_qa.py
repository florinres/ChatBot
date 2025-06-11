import os
import json
import requests
import time

input_file = "rezultat.txt"
output_file = "instruction_dataset.jsonl"
scrise_deja = set()

# 🧪 Încarcă deja existentele pentru a evita duplicate
if os.path.exists(output_file):
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                scrise_deja.add(item["output"])
            except:
                pass

# 🔧 Funcția de generare întrebare folosind Ollama + Mistral
def generate_question_locally(paragraph):
    prompt = (
        f"Generează o întrebare potrivită în limba română pentru următorul paragraf:\n\n"
        f"{paragraph}\n\n"
        f"Întrebare:"
    )

    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False
            }
        )
        result = response.json()
        return result.get("response", "").strip()
    except Exception as e:
        print(f"[EROARE Mistral] {e}")
        return "Întrebare nereușită"

# 📤 Scriem incremental rezultatele
with open(output_file, "a", encoding="utf-8") as f_out:
    if os.path.exists(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Sparge textul în paragrafe pe baza liniilor goale
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        for idx, paragraph in enumerate(paragraphs):
            if len(paragraph) < 30 or paragraph in scrise_deja:
                continue

            question = generate_question_locally(paragraph)

            example = {
                "instruction": question,
                "input": "",
                "output": paragraph
            }

            print(f"\n📄 Paragraf {idx + 1}")
            print(f"❓ Întrebare: {question}")
            print(f"📝 Răspuns:\n{paragraph}\n")

            f_out.write(json.dumps(example, ensure_ascii=False) + "\n")
            f_out.flush()

            scrise_deja.add(paragraph)
            time.sleep(0.3)
    else:
        print(f"[EROARE] Fișierul '{input_file}' nu a fost găsit.")
