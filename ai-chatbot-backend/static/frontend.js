// In your frontend JavaScript
const chatbotBackendUrl = 'https://ai-chatbot-service-94128419367.us-central1.run.app'; // REPLACE WITH YOUR ACTUAL CLOUD RUN URL

async function sendQuestionToAI() {
    const questionInput = document.getElementById('questionInput');
    const question = questionInput.value;
    if (!question) return;

    // Display user question in chat history
    // ...

    try {
        const response = await fetch(chatbotBackendUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question: question })
        });

        const data = await response.json();
        if (response.ok) {
            // Display AI's answer in chat history
            // ...
        } else {
            // Handle error
            console.error("Error from AI backend:", data.error);
            // Display error to user
        }
    } catch (error) {
        console.error("Network or fetch error:", error);
        // Display network error to user
    } finally {
        questionInput.value = ''; // Clear input
    }
}