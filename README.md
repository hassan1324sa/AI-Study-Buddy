AI Study Buddy

AI Study Buddy is a web-based application designed to enhance your learning experience by leveraging artificial intelligence. Upload PDF documents (up to 50MB), ask questions about their content, and generate interactive quizzes to test your knowledge. Built with a modern tech stack, this project combines a user-friendly interface with powerful AI capabilities to make studying smarter and more efficient.
Table of Contents

Features
Tech Stack
Installation
Usage
Contributing
License
Contact

Features



Feature
Description



PDF Upload & Indexing
Upload PDF files (up to 50MB) and index their content for quick retrieval and querying.


AI-Powered Q&A
Ask questions about uploaded documents, with AI providing accurate answers and source references (including page numbers).


Interactive Quizzes
Generate customizable quizzes based on a specified topic or uploaded content, with instant feedback and detailed results.


User Authentication
Secure login and registration system to manage user sessions and data.


Responsive Design
A modern, responsive UI with smooth animations, compatible with desktop and mobile devices.


Drag-and-Drop Upload
Intuitive drag-and-drop interface for seamless PDF uploads.


Tech Stack

Frontend: HTML, CSS, JavaScript
Backend: FastAPI (Python)
AI Integration: Powered by Hugging Face (hosting and potential model integration)
Styling: Font Awesome for icons, custom CSS with gradient themes
Deployment: Hosted on Hugging Face Spaces
Other: Fetch API for asynchronous requests, FormData for file uploads

Installation
To run AI Study Buddy locally, follow these steps:
Prerequisites

Python 3.8+
Node.js (for local frontend testing, optional)
Git
A Hugging Face account (for API access, if using their models)

Steps

Clone the Repository:
git clone https://github.com/hassan1324sa/AI-Study-Buddy.git
cd AI-Study-Buddy


Set Up the Backend:

Install Python dependencies:pip install -r requirements.txt


Start the FastAPI server:uvicorn app:app --reload


The backend will run at http://localhost:8000.


Serve the Frontend:

Copy the index.html file to a local server directory or use a simple HTTP server:python -m http.server 8080


Open http://localhost:8080 in your browser.


Optional: Deploy to Hugging Face Spaces:

Create a new Space on Hugging Face.
Push the repository to your Space, ensuring app.py, index.html, requirements.txt, and Dockerfile are included.
Configure the Space to use the FastAPI app.



Usage

Access the App:

Visit the deployed app at https://hassan123123-ai-study-buddy.hf.space or your local server (http://localhost:8080).
Sign in or create an account to access the main features.


Upload PDF:

Navigate to the "Upload Files" tab.
Drag and drop a PDF file (≤50MB) or click to select one.
Click "Upload File" to index the PDF for querying.


Ask Questions:

Go to the "Chat" tab.
Type a question related to your uploaded PDFs.
The AI will respond with answers and cite sources (e.g., document and page number).


Generate Quizzes:

In the "Quiz" tab, enter a topic (e.g., "Organic Chemistry") and the number of questions (1–50).
Click "Generate Quiz" to create an interactive quiz.
Select answers, check results, and review detailed feedback.



Contributing
Contributions are welcome! To contribute to AI Study Buddy:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Make your changes and commit (git commit -m "Add your feature").
Push to your branch (git push origin feature/your-feature).
Open a pull request with a detailed description of your changes.

Please follow the Code of Conduct and ensure your code adheres to the project's style guidelines.
License
Distributed under the MIT License. See LICENSE for more information.
Contact

Author: Hassan
GitHub: hassan1324sa
Project Link: AI-Study-Buddy
Live Demo: Hugging Face Space

Feel free to open an issue or reach out for questions, feedback, or collaboration!
