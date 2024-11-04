## Installation

1. Clone the repository:
    git clone https://github.com/abhi-gowdaa/aiplanet-backend.git 
    cd to main folder


2. Create a virtual environment and activate it:(optional)
    python -m venv venv
    source venv/bin/activate  
    `venv\Scripts\activate`


3. Install the required dependencies:
    pip install -r requirements.txt


## Configuration

1. Create a `.env` file in the project root and add your Gemini API key:

    GEMINI_API_KEY=your_gemini_api_key_here
 

## Running the Application

1. Start the FastAPI server:
    " uvicorn main:app --reload "
 
2. The server will start on `http://127.0.0.1:8000`.

 
