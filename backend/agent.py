from agents import Agent, Runner, Session
from typing import List, Dict, Optional
import json
import os
import asyncio
from pydantic import BaseModel

# Define data structure for storing conversation and feedback
class ConversationData(BaseModel):
    prompt: str
    response: str
    user_feedback: Optional[int] = None  # Feedback score (1-5)
    timestamp: str

class RLHFAgent:
    def __init__(self, session_id: str, data_file: str = "conversation_data.json"):
        self.data_file = data_file
        self.session = Session(session_id=session_id)
        
        # Initialize conversation data storage
        self.conversation_history: List[ConversationData] = []
        if os.path.exists(data_file):
            with open(data_file, 'r') as f:
                data = json.load(f)
                self.conversation_history = [ConversationData(**item) for item in data]

        # Define agents
        self.conversation_agent = Agent(
            name="ConversationAgent",
            instructions="You are a helpful assistant engaging in back-and-forth conversation. "
                        "Provide clear, concise, and helpful responses to user prompts. "
                        "Encourage users to provide feedback on your responses (1-5 scale).",
            model="gpt-4o"
        )
        
        self.evaluation_agent = Agent(
            name="EvaluationAgent",
            instructions="You help users evaluate previous responses. "
                        "Provide a summary of past conversations and ask for feedback (1-5). "
                        "If asked, analyze feedback trends and suggest improvements.",
            model="gpt-4o"
        )

    async def save_conversation(self, prompt: str, response: str, timestamp: str):
        """Save conversation data to file."""
        conversation = ConversationData(
            prompt=prompt,
            response=response,
            timestamp=timestamp
        )
        self.conversation_history.append(conversation)
        with open(self.data_file, 'w') as f:
            json.dump([item.dict() for item in self.conversation_history], f, indent=2)

    async def save_feedback(self, index: int, feedback: int):
        """Save user feedback for a specific conversation."""
        if 0 <= index < len(self.conversation_history) and 1 <= feedback <= 5:
            self.conversation_history[index].user_feedback = feedback
            with open(self.data_file, 'w') as f:
                json.dump([item.dict() for item in self.conversation_history], f, indent=2)
            return True
        return False

    async def get_conversation_history(self) -> List[Dict]:
        """Retrieve conversation history."""
        return [item.dict() for item in self.conversation_history]

    async def run_conversation(self, user_input: str) -> str:
        """Handle back-and-forth conversation."""
        result = await Runner.run(
            agent=self.conversation_agent,
            input=user_input,
            session=self.session
        )
        response = result.final_output
        from datetime import datetime
        await self.save_conversation(user_input, response, datetime.now().isoformat())
        return f"{response}\n\nPlease provide feedback on this response (1-5, where 5 is best):"

    async def evaluate_responses(self, user_input: str) -> str:
        """Allow users to evaluate past responses or analyze feedback."""
        if user_input.lower().startswith("feedback"):
            try:
                # Expecting input like "feedback <index> <score>"
                _, index, score = user_input.split()
                index, score = int(index), int(score)
                if await self.save_feedback(index, score):
                    return f"Feedback saved for conversation #{index}. Score: {score}/5"
                return "Invalid index or score. Please use index from 0 to {} and score from 1 to 5.".format(len(self.conversation_history) - 1)
            except ValueError:
                return "Please use format: 'feedback <index> <score>' (e.g., 'feedback 0 4')"
        
        elif user_input.lower() == "analyze":
            history = await self.get_conversation_history()
            feedback_scores = [item['user_feedback'] for item in history if item['user_feedback'] is not None]
            if not feedback_scores:
                return "No feedback available yet."
            avg_score = sum(feedback_scores) / len(feedback_scores)
            suggestions = "Keep up the good work!" if avg_score >= 4 else "Consider improving response clarity and relevance."
            return f"Feedback Analysis:\nAverage Score: {avg_score:.2f}/5\nNumber of Feedbacks: {len(feedback_scores)}\nSuggestions: {suggestions}"

        else:
            # Show conversation history for evaluation
            history = await self.get_conversation_history()
            if not history:
                return "No conversation history yet. Start a conversation first!"
            output = "Conversation History:\n"
            for i, item in enumerate(history):
                output += f"#{i} Prompt: {item['prompt']}\nResponse: {item['response']}\nFeedback: {item['user_feedback'] or 'Not provided'}\n\n"
            output += "To provide feedback, use: 'feedback <index> <score>' (e.g., 'feedback 0 4')\nTo analyze feedback trends, type: 'analyze'"
            return output

async def main():
    agent = RLHFAgent(session_id="rlhf_session_001")
    print("Welcome to the RLHF Agent! Type 'evaluate' to review conversations, 'analyze' for feedback trends, or enter a prompt to chat.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        elif user_input.lower() in ["evaluate", "analyze"] or user_input.lower().startswith("feedback"):
            response = await agent.evaluate_responses(user_input)
        else:
            response = await agent.run_conversation(user_input)
        print(f"Agent: {response}")

if __name__ == "__main__":
    asyncio.run(main())