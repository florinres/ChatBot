from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
from load_faiss import get_vector_store, find_relevant_urls

print("🔧 Inițializare server model...")

# Incarcă modelul TinyLLaMA + adapter PEFT
base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_path = "D:/chatbot/adapter"

print("🔄 Încărcare model și tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(base_model_id).to("cuda")
model = PeftModel.from_pretrained(base_model, adapter_path).merge_and_unload().to("cuda")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0, max_new_tokens=256)

print("✅ Model încărcat și pregătit.")

# Încarcă FAISS vector store o singură dată
print("🔍 Construim vector store FAISS...")
vector_store = get_vector_store()
if not vector_store:
    raise RuntimeError("❌ Nu s-au putut încărca documentele în FAISS.")

print("✅ Vector store creat.")

# Flask app pentru primire întrebări
app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    question = data.get("prompt", "")

    if not question.strip():
        return jsonify({"error": "Promptul este gol."}), 400

    # Căutare în vector store
    try:
        relevant_docs = vector_store.similarity_search(question, k=3)
        context = "\n".join([doc.page_content for doc in relevant_docs])
    except Exception as e:
        return jsonify({"error": f"Eroare la căutare: {e}"}), 500

    if not context.strip():
        return jsonify({"answer": "❗ Nu am găsit informații relevante în documente."})

    # Generează răspuns
    prompt = f"<|user|>\nContext:\n{context}\n\nÎntrebare: {question}\nTe rog să răspunzi în limba română.\n<|assistant|>"
    try:
        output = pipe(prompt)[0]['generated_text'].replace(prompt, '').strip()
    except Exception as e:
        return jsonify({"error": f"Eroare la generare: {e}"}), 500

    # Caută linkuri utile
    urls = find_relevant_urls(question, context)
    if urls:
        output += "\n\n📎 Linkuri utile:"
        for url in urls[:3]:
            output += f"\n• {url}"

    return jsonify({"answer": output})


if __name__ == "__main__":
    app.run(port=5001)
