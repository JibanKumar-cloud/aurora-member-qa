#### Aurora Member Q&A Retrieval System
## Semantic Search + OpenAI Reasoning (FAISS Persistent Index)
# 1) Install all dependencies

Run the following command:
pip install -r requirements.txt

# 2) Running the API locally

Start the API server:
uvicorn app.main:app --reload

# Before doing that, set your local environment variables:

export OPENAI_API_KEY="your-openai-key"
export HF_TOKEN="your-huggingface-token"
export MESSAGES_API_URL="https://your-messages-api/messages/"


The system uses a prebuilt FAISS index stored in data/:

data/faiss.index
data/messages_meta.json

If these files exist, the system loads them instantly.

If not, it fetches messages from the MESSAGES_API_URL, builds embeddings with MiniLM, creates a FAISS index, and persists them.

3) Core Files
# a) retriever.py

Handles:

Loading the FAISS index + metadata
Rebuilding the index if missing
Semantic search (semantic_search())
Fetching raw messages
Persisting index to disk


# b) main.py

# Contains the FastAPI server:

/ask endpoint
# Accepts two parameters :

q → user question
backend=openai → use OpenAI reasoning
backend=semantic → only return retrieved messages
By default: backend=openai

# Pipeline:
Retrieve top-k messages via FAISS
If backend=openai → send retrieved messages to the LLM
Return final structured response


## Cloud Deployment
# Deploy using:

gcloud run deploy aurora-member-qa-api \
  --source . \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 1Gi \
  --set-env-vars MESSAGES_API_URL="https://your-api/messages/" \
  --set-secrets OPENAI_API_KEY=openai-api-key:latest,HF_TOKEN=huggingface-token:latest


# Cloud Run will:

Load the FAISS index from the container
Never call HuggingFace unless rebuilding
Never call the /messages API unless index missing

### Running Locally:
## Testing the API

# Example:
curl "http://127.0.0.1:8000/ask?q=When%20is%20Sophia%20Al-Farsi%20going%20to%20Paris%3F&backend=openai"


or semantic search only:

curl "http://127.0.0.1:8000/ask?q=Where%20does%20Layla%20Kawaguchi%20want%20to%20book%20a%20villa%3F&backend=semantic"

### Sample of data:
{"total":3349,"items":[{"id":"b1e9bb83-18be-4b90-bbb8-83b7428e8e21","user_id":"cd3a350e-dbd2-408f-afa0-16a072f56d23","user_name":"Sophia Al-Farsi","timestamp":"2025-05-05T07:47:20.159073+00:00","message":"Please book a private jet to Paris for this Friday."},{"id":"609ba052-c9e7-49e6-8b62-061eb8785b63","user_id":"e35ed60a-5190-4a5f-b3cd-74ced7519b4a","user_name":"Fatima El-Tahir","timestamp":"2024-11-14T20:03:44.159235+00:00","message":"Can you confirm my dinner reservation at The French Laundry for four people tonight?"},{"id":"44be0607-a918-40fa-a122-b2435fe54f3e","user_id":"23103ae5-38a8-4d82-af82-e9942aa4aefb","user_name":"Armand Dupont","timestamp":"2025-03-09T02:25:23.159256+00:00","message":"I need two tickets to the opera in Milan this Saturday."}]}

### API has been deployed in google cloud:
## Testing the API
1) "id": "b1e9bb83-18be-4b90-bbb8-83b7428e8e21"
    message: "Please book a private jet to Paris for this Friday."
    question: When is Sophia Al Farsi going to Paris?
    Local API: http://127.0.0.1:8000/ask?q=When%20is%20Sophia%20Al%20Farsi%20going%20to%20Paris%3F&backend=openai
    Deployed API: https://aurora-member-qa-api-1067461909384.us-central1.run.app/ask?q=When%20is%20Sophia%20Al%20Farsi%20going%20to%20Paris%3F&backend=openai

2) "id": "609ba052-c9e7-49e6-8b62-061eb8785b63"
    "message": "Can you confirm my dinner reservation at The French Laundry for four people tonight?"
    question: Where is fatima El Tahir dinner reservation and for how many people?
    Local API: http://127.0.0.1:8000/ask?q=Where%20is%20Fatima%20El%20Tahir%20dinner%20reservation%20and%20for%20how%20many%20people%3F&backend=openai
    Deployed API: https://aurora-member-qa-api-1067461909384.us-central1.run.app/ask?q=Where%20is%20Fatima%20El%20Tahir%20dinner%20reservation%20and%20for%20how%20many%20people%3F&backend=openai

3) "id": "43d8a12e-4fdb-4c82-8a78-f7dfff583b9f"
    "message": "Please remember I prefer aisle seats during my flights."
    question: "What kind of airplane seat does Layla Kawaguchi prefer?"
    Local API: http://127.0.0.1:8000/ask?q=What%20kind%20of%20airplane%20seat%20does%20Layla%20Kawaguchi%20prefer%3F&backend=openai
    Deployed API: https://aurora-member-qa-api-1067461909384.us-central1.run.app/ask?q=What%20kind%20of%20airplane%20seat%20does%20Layla%20Kawaguchi%20prefer%3F&backend=openai

4) "id": "4df0ad3b-d73a-45fa-81b5-29f803d54783"
    "message": "Book a villa in Santorini for the first week of December."
    question: "Where does Layla Kawaguchi want to book a villa?"
    Local API: http://127.0.0.1:8000/ask?q=Where%20does%20Layla%20Kawaguchi%20want%20to%20book%20a%20villa%3F&backend=openai
    Deployed API: https://aurora-member-qa-api-1067461909384.us-central1.run.app/ask?q=Where%20does%20Layla%20Kawaguchi%20want%20to%20book%20a%20villa%3F&backend=openai

5) "id": "c84b1706-5cc6-45b3-aa7e-3b6d83ad0bce"
    "message": "Could you confirm the spa appointment for next Tuesday?"
    question: "What appointment does Sophia Al Farsi want confirmed for next Tuesday?"
    Local API: http://127.0.0.1:8000/ask?q=What%20appointment%20does%20Sophia%20Al%20Farsi%20want%20confirmed%20for%20next%20Tuesday%3F&backend=openai
    Deployed API: https://aurora-member-qa-api-1067461909384.us-central1.run.app/ask?q=What%20appointment%20does%20Sophia%20Al%20Farsi%20want%20confirmed%20for%20next%20Tuesday%3F&backend=openai

### Sample Output:
https://aurora-member-qa-api-1067461909384.us-central1.run.app/ask?q=Where%20is%20Fatima%20El%20Tahir%20dinner%20reservation%20and%20for%20how%20many%20people%3F&backend=openai
output: 
{"answer":"Fatima El-Tahir's dinner reservation is at The French Laundry for four people tonight.","reasoning":{"backend":"openai","explanation":"The message with id 609ba052-c9e7-49e6-8b62-061eb8785b63 explicitly mentions a dinner reservation at The French Laundry for four people. Although another message mentions reserving a private dining room for ten at Gaggan in Bangkok, it does not specify whether this was confirmed or relates to the same reservation, so the confirmed dinner reservation is at The French Laundry for four."}}

https://aurora-member-qa-api-1067461909384.us-central1.run.app/ask?q=Where%20is%20Fatima%20El%20Tahir%20dinner%20reservation%20and%20for%20how%20many%20people%3F&backend=semantic

output:
{"answer":"Based on the messages, the best matching information is:\n\"Can you confirm my dinner reservation at The French Laundry for four people tonight?\"","reasoning":{"backend":"semantic","reasoning":"I ranked all messages by semantic similarity to the question and selected the top one as the most relevant."}}

ALTERNATIVES:
We can definately use fine tuned model as agent if we don't want to share our data to third party vendor like OpenAI.

#### Notes

If FAISS index exists, no transformer model is downloaded.
Memory in Cloud Run must be > 512 MB because MiniLM loads at startup.
_STORE is a global singleton → efficient for high concurrency.
All secrets are stored in Google Secret Manager, not in code.