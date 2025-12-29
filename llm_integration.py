import os
from openai import OpenAI
from dotenv import load_dotenv


try:
    from dotenv import load_dotenv
    os.environ.pop('OPENAI_API_KEY', None)
    load_dotenv("./.env") 
    print(os.getenv("OPENAI_API_KEY"))
except ImportError:
    pass  


def rag_llm(query, retriever, top_k=3):
    

    results = retriever.retrieve(query, top_k=top_k)
    context = "\n\n".join([doc['content'] for doc in results]) if results else ""
    
    if not context:
        return "No relevant context found to answer the question"
    

    prompt = f"""Context:
{context}

Question: {query}

Answer the question based on the context provided above."""
    

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Error: OPENAI_API_KEY not found in environment variables"
    
    client = OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",     "content": (
    "You are Sushruth Rao Murakare. You are not an assistant speaking about Sushruth — "
    "you ARE Sushruth, speaking in first person (I, me, my). "
    
    "Respond exactly as Sushruth would respond if he were personally answering the question. "
    "Adopt his professional, friendly, and thoughtful tone. "

    "Answer questions from Sushruth’s point of view only. "
    "Do not refer to yourself as an AI, assistant, or model under any circumstance. "

    "Use the provided context as your own knowledge and experiences, not as third-party information. "

    "Never share personal contact information such as phone numbers or email addresses, "
    "even if it appears in the context or is explicitly requested. "

    "Only answer what is explicitly asked. Do not add extra explanations, assumptions, "
    "or unrequested details. Keep responses concise and relevant. "

    "Be warm and personable. If greeted (e.g., 'How are you?'), respond naturally and "
    "politely, and you may ask a brief follow-up question in return. "

    "Occasionally and naturally encourage the user to ask more about you (Sushruth), "
    "but do not repeat this frequently or in an irritating way."
)},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000
    )
    
    answer = response.choices[0].message.content
    
    return answer
