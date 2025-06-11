import os
import re

INPUT_FOLDER = "output"
OUTPUT_FILE = "rezultat.txt"
MIN_PARAGRAPH_LENGTH = 40

def este_prostie(text):
    if re.search(r"(.)\1{4,}", text):
        print(f"[!] Eliminat pentru repetitii: {text[:60]}")
        return True
    if re.search(r"[^\w\s.,;!?()\"\']+", text):
        print(f"[!] Eliminat pentru simboluri suspecte: {text[:60]}")
        return True
    if len(re.findall(r"\b\w{15,}\b", text)) > 2:
        print(f"[!] Eliminat pentru cuvinte prea lungi: {text[:60]}")
        return True
    return False

def curata_text(text):
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # elimina caractere non-ASCII
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def proceseaza_fisiere(input_folder):
    rezultate = []
    fisiere = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
    print(f"[i] Găsite {len(fisiere)} fișiere în folderul '{input_folder}'.")

    for fname in fisiere:
        path = os.path.join(input_folder, fname)
        with open(path, 'r', encoding='utf-8') as f:
            continut = f.read()
            continut = curata_text(continut)
            paragrafe = continut.split('. ')  # separare simplă pe propoziții/paragrafe
            print(f"[i] Fișier '{fname}' are {len(paragrafe)} paragrafe inițiale.")

            for p in paragrafe:
                p = p.strip()
                if len(p) >= MIN_PARAGRAPH_LENGTH and not este_prostie(p):
                    rezultate.append(p)
                else:
                    print(f"[!] Paragraf respins ({len(p)} caractere): {p[:60]}")
    return rezultate

def scrie_rezultat(rezultate, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for paragraf in rezultate:
            f.write(paragraf + "\n\n")

if __name__ == "__main__":
    paragrafe_curate = proceseaza_fisiere(INPUT_FOLDER)
    scrie_rezultat(paragrafe_curate, OUTPUT_FILE)
    print(f"[✓] Procesare completă. {len(paragrafe_curate)} paragrafe salvate în '{OUTPUT_FILE}'.")
