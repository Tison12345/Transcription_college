
from llama_cpp import Llama

MODEL_PATH = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf"

print("Loading Qwen2.5 model...")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=6,
    n_gpu_layers=0,
    verbose=False
)

print("Qwen2.5 loaded successfully!")



SYSTEM_PROMPT = (
    "You are a warm, professional patient follow-up assistant on a phone call. "
    "Your ONLY task is to acknowledge what the patient just said, "
    "in the context of the question they were asked. "
    "Respond with exactly one short sentence, spoken naturally and warmly. "
    "Focus on the patient's feelings and experience. "
    "Never express your own feelings, and never say things like "
    "'I feel', 'it feels good', or 'thank you for sharing'. "
    "Vary your wording and avoid generic filler phrases. "
    "Do not ask questions and never use a question mark. "
    "Do not give medical advice, diagnoses, recommendations, "
    "or exercise instructions, and do not tell the patient what to do. "
    "Do not introduce a new topic. "
    "Do not invent details the patient has not mentioned. "
    "Interpret short answers such as 'yes' or 'no' according to the question. "
    "If the patient reports pain or discomfort, acknowledge their experience "
    "with empathy. "
    "If the patient reports no pain, respond positively. "
    "If the patient reports improvement or good news, respond positively. "
    "If the patient has not done an exercise, respond without any judgment. "
    "If the patient confirms it is a good time to talk, simply acknowledge it "
    "briefly and politely.\n\n"
    "Examples:\n"
    "Question: Are you experiencing any pain or discomfort? "
    "Patient: No, I am not experiencing any pain or discomfort.\n"
    "Reply: I'm glad to hear you're not having any pain.\n"
    "Question: Are you experiencing any pain or discomfort? "
    "Patient: I have mild ankle pain.\n"
    "Reply: I'm sorry you're dealing with that ankle pain.\n"
    "Question: Is this a good time to talk? Patient: Yes, this is a good time.\n"
    "Reply: Wonderful, thank you for making the time.\n"
    "Question: Have you been able to do your prescribed toe exercise? "
    "Patient: No, I have not done it.\n"
    "Reply: That's all right, thank you for letting me know."
)


def generate_acknowledgment(patient_response, question=None):

    if question:
        user_content = (
            f"Question asked: {question}\n"
            f"Patient's answer: {patient_response}"
        )
    else:
        user_content = f"Patient's answer: {patient_response}"

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_content
        }
    ]

    result = llm.create_chat_completion(
        messages=messages,
        max_tokens=100,
        temperature=0.3
    )

    return result["choices"][0]["message"]["content"].strip()

if __name__ == "__main__":

    test_responses = [
        "I am feeling a little pain in my ankle.",
        "I am feeling much better today.",
        "I have not been doing my exercises regularly."
    ]

    for patient_response in test_responses:
        print("\nPatient:", patient_response)

        ai_response = generate_acknowledgment(patient_response)

        print("AI:", ai_response)
