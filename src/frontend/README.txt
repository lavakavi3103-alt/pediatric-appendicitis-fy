Pediatric AI Diagnosis frontend prototype

1. Extract the frontend folder into your project root.
2. Start your existing FastAPI backend:
   .\.venv\Scripts\python.exe src\16_fastapi_backend.py
3. Open frontend\index.html in a browser.

This UI calls /patient/validate and /ultrasound/validate only.
It does not claim to make AI predictions. Genuine inference requires matching
saved preprocessing objects and model pipelines to be added to the backend.
