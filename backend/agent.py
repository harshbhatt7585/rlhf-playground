import openai
from typing import List, Dict, Optional
import json
import os
import asyncio
from datetime import datetime
from pydantic import BaseModel

# Define data structure for storing conversation and feedback
class ConversationData(BaseModel):
    prompt: str
    response: str
    user_feedback: Optional[int] = None  # Feedback score (1-5)
    timestamp: str

class RLHFAgent:
    def __init__(self, api_key: str, model: str = "gpt-4o", data_file: str = "conversation_data.json"):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
        self.data_file = data_file
        
        # Initialize conversation data storage
        self.conversation_history: List[ConversationData] = []
        if os.path.exists(data_file):
            with open(data_file, 'r') as f:
                data = json.load(f)
                self.conversation_history = [ConversationData(**item) for item in data]

        # System messages for different agent roles
        self.conversation_system_message = (
            "You are a helpful assistant engaging in back-and-forth conversation. "
            "Provide clear, concise, and helpful responses to user prompts. "
            "Be friendly and informative while keeping responses focused and relevant."
        )
        
        self.evaluation_system_message = (
            "You help users evaluate previous responses and analyze feedback trends. "
            "Provide clear summaries of past conversations and meaningful insights "
            "based on feedback patterns. Suggest specific improvements when appropriate."
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
        """Handle back-and-forth conversation using OpenAI API."""
        try:
            # Build conversation context from recent history
            messages = [{"role": "system", "content": self.conversation_system_message}]
            
            # Add recent conversation context (last 5 exchanges)
            recent_history = self.conversation_history[-5:] if len(self.conversation_history) > 5 else self.conversation_history
            for conv in recent_history:
                messages.append({"role": "user", "content": conv.prompt})
                messages.append({"role": "assistant", "content": conv.response})
            
            # Add current user input
            messages.append({"role": "user", "content": user_input})
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            assistant_response = response.choices[0].message.content
            timestamp = datetime.now().isoformat()
            
            # Save the conversation
            await self.save_conversation(user_input, assistant_response, timestamp)
            
            return f"{assistant_response}\n\nPlease provide feedback on this response (1-5, where 5 is best):"
            
        except Exception as e:
            return f"Error generating response: {str(e)}"

    async def evaluate_responses(self, user_input: str) -> str:
        """Allow users to evaluate past responses or analyze feedback."""
        if user_input.lower().startswith("feedback"):
            try:
                # Expecting input like "feedback <index> <score>"
                parts = user_input.split()
                if len(parts) != 3:
                    return "Please use format: 'feedback <index> <score>' (e.g., 'feedback 0 4')"
                
                _, index_str, score_str = parts
                index, score = int(index_str), int(score_str)
                
                if await self.save_feedback(index, score):
                    return f"Feedback saved for conversation #{index}. Score: {score}/5"
                return f"Invalid index or score. Please use index from 0 to {len(self.conversation_history) - 1} and score from 1 to 5."
            except ValueError:
                return "Please use format: 'feedback <index> <score>' (e.g., 'feedback 0 4')"
        
        elif user_input.lower() == "analyze":
            return await self.analyze_feedback()
        
        else:
            # Show conversation history for evaluation
            history = await self.get_conversation_history()
            if not history:
                return "No conversation history yet. Start a conversation first!"
            
            output = "Conversation History:\n" + "="*50 + "\n"
            for i, item in enumerate(history):
                feedback_str = f"{item['user_feedback']}/5" if item['user_feedback'] else "Not provided"
                output += f"#{i} | {item['timestamp'][:19]}\n"
                output += f"Prompt: {item['prompt']}\n"
                output += f"Response: {item['response']}\n"
                output += f"Feedback: {feedback_str}\n"
                output += "-" * 50 + "\n"
            
            output += "\nCommands:\n"
            output += "• 'feedback <index> <score>' - Provide feedback (e.g., 'feedback 0 4')\n"
            output += "• 'analyze' - View feedback trends and suggestions\n"
            return output

    async def analyze_feedback(self) -> str:
        """Analyze feedback patterns and provide insights."""
        history = await self.get_conversation_history()
        feedback_scores = [item['user_feedback'] for item in history if item['user_feedback'] is not None]
        
        if not feedback_scores:
            return "No feedback available yet. Please provide feedback on some conversations first."
        
        # Basic statistics
        avg_score = sum(feedback_scores) / len(feedback_scores)
        max_score = max(feedback_scores)
        min_score = min(feedback_scores)
        total_conversations = len(history)
        feedback_rate = (len(feedback_scores) / total_conversations) * 100
        
        # Score distribution
        score_dist = {i: feedback_scores.count(i) for i in range(1, 6)}
        
        # Generate insights using OpenAI
        try:
            analysis_prompt = f"""
            Analyze this feedback data and provide insights:
            
            Total Conversations: {total_conversations}
            Feedback Rate: {feedback_rate:.1f}%
            Average Score: {avg_score:.2f}/5
            Score Range: {min_score} - {max_score}
            Score Distribution: {score_dist}
            
            Recent feedback scores: {feedback_scores[-10:]}
            
            Provide specific, actionable suggestions for improvement based on this data.
            """
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.evaluation_system_message},
                    {"role": "user", "content": analysis_prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            ai_insights = response.choices[0].message.content
            
            result = f"📊 Feedback Analysis\n" + "="*50 + "\n"
            result += f"Total Conversations: {total_conversations}\n"
            result += f"Feedback Rate: {feedback_rate:.1f}%\n"
            result += f"Average Score: {avg_score:.2f}/5\n"
            result += f"Score Range: {min_score} - {max_score}\n"
            result += f"Score Distribution: {score_dist}\n\n"
            result += f"🤖 AI Insights:\n{ai_insights}"
            
            return result
            
        except Exception as e:
            # Fallback to basic analysis if AI analysis fails
            suggestions = "Keep up the good work!" if avg_score >= 4 else "Consider improving response clarity and relevance."
            return f"📊 Feedback Analysis\n" + "="*50 + "\n" + f"Average Score: {avg_score:.2f}/5\nNumber of Feedbacks: {len(feedback_scores)}\nSuggestions: {suggestions}"

async def main():
    # Initialize the agent
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set your OPENAI_API_KEY environment variable")
        return
    
    agent = RLHFAgent(api_key=api_key)
    
    print("🤖 Welcome to the RLHF Agent!")
    print("Commands:")
    print("• Type a message to start a conversation")
    print("• Type 'evaluate' to review conversation history")
    print("• Type 'analyze' to see feedback trends")
    print("• Type 'feedback <index> <score>' to rate a response")
    print("• Type 'exit' to quit")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\n🔵 You: ").strip()
            
            if user_input.lower() == "exit":
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
                
            if user_input.lower() in ["evaluate", "analyze"] or user_input.lower().startswith("feedback"):
                response = await agent.evaluate_responses(user_input)
            else:
                response = await agent.run_conversation(user_input)
            
            print(f"\n🤖 Agent: {response}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())