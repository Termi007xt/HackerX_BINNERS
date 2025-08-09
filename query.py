from pinecone_db import embed_chunks, search
from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()


def query_llm(json_chunks, queries):
    client = Groq(
        api_key=os.environ['GROQ_API_KEY'],
    )

    chunks, embeddings = embed_chunks(json_chunks)

    # Batch query to LLM
    context = ''
    for query in queries:
        top_chunks = search(query, chunks, embeddings)

        context += '\n\n'.join(top_chunks)

    prompt = f"""You are an expert legal assistant.
    Use the following context to answer the following questions.
    Separate each answer with a '|'. Do not repeat the questions in the answer.
    If it's not answerable from the text, say "Not found in document"
    
    Context:
    {context}
    
    Question:
    {queries}
    
    Answer: """

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"{prompt}",
            }
        ],
        model="meta-llama/llama-4-scout-17b-16e-instruct",
    )
    answers = chat_completion.choices[0].message.content

    # Postprocessing
    answers = answers.split('|')
    answers = [answer.strip() for answer in answers if answer.strip()]
    # with open('answers.txt', 'a', encoding='utf-8') as file:
    #     file.write('New ans start here' + str(answers) + '\n')

    ans = {"answers": answers}

    return ans