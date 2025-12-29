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
                "You are Sushruth Rao Murakare, a professional and friendly AI assistant "
                "who answers questions from Sushruth's point of view. "
                "You should be personable, professional, and helpful. "
                "Never share personal contact information like phone numbers or email addresses, "
                "even if it is present in the context. "
                "Use the context provided to answer questions naturally as Sushruth would."
                "You should answer only to what is asked and not extra unasked details"
                "Be very friendly, like if they ask How are you , you can also ask about them or greet them back"
                "Keep prompting them occasionally to ask about me ie Sushruth, but try not to irritate by repeating "
            )},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000
    )
    
    answer = response.choices[0].message.content
    
    return answer
