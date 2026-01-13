---
description: Guide for using the PDF Chat feature
---

# PDF Chat Workflow

This workflow explains how to interact with your PDF documents using the RAG-enabled chat interface.

## 1. Navigate to PDF Chat
Launch the app and select the **PDF Chat** page from the sidebar navigation.

## 2. Upload a PDF
- Click on the **Upload PDF files** button.
- Select the PDF document you want to explore.
- Wait for the "Processing..." spinner to complete. The app will generate embeddings and store them in a local vector database.

## 3. Quick Actions
You can use the predefined buttons for common tasks:
- **Summarize Document**: Get a high-level overview.
- **Generate Questions**: See what the AI thinks are the most relevant questions.
- **Key Takeaways**: Extract the main points quickly.

## 4. Ask Custom Questions
- Use the chat input field at the bottom to ask specific questions.
- The AI will use both previous conversation history and relevant sections of the PDF to answer.

## 5. View History
The chat history is persistent across sessions and can be found in the `./chats/chat.json` file.
