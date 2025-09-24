from fastapi import FastAPI
# from api.chatbot.rag import app as rag_router
from chatbot.app.api.chatbot.rag import rag_router
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(openapi_url="/api/openapi.json", docs_url="/api/docs")

# Define allowed origins
origins = [
    "http://localhost:3000",  # React, Vue, Angular frontend (local development)
    "http://127.0.0.1:3000",
]

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Specifies allowed frontend domains
    allow_credentials=True,  # Allows cookies/auth
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

app.include_router(rag_router, prefix="/api")

@app.get("/")
def read_root():
    return {"message": "Chatbot is running"}