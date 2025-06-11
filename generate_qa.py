import os
import json
import requests
import time

input_folder = "output"
output_file = "instruction_dataset.jsonl"
scrise_deja = set()

# 🧪 Încarcă deja existentele (pentru a evita duplicate dacă rulezi din nou)
if os.path.exists(output_file):
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                scrise_deja.add(item["output"])
            except:
                pass

# 🔧 Funcția care generează întrebarea local, prin Ollama + Mistral
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

# 📤 Deschidem fișierul pentru scriere incrementală
with open(output_file, "a", encoding="utf-8") as f_out:
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(input_folder, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            # Sparge în paragrafe
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

            for idx, paragraph in enumerate(paragraphs):
                if len(paragraph) < 30 or paragraph in scrise_deja:
                    continue  # evită paragrafe goale, scurte sau deja procesate

                # Generează întrebarea
                question = generate_question_locally(paragraph)

                # Creează obiectul
                example = {
                    "instruction": question,
                    "input": "",
                    "output": paragraph
                }

                # Afișează în consolă
                print(f"\n📄 {filename} | Paragraf {idx+1}")
                print(f"❓ Întrebare: {question}")
                print(f"📝 Răspuns:\n{paragraph}\n")

                # Scrie în fișier
                f_out.write(json.dumps(example, ensure_ascii=False) + "\n")
                f_out.flush()

                scrise_deja.add(paragraph)
                time.sleep(0.7)  # pauză scurtă între requesturi
