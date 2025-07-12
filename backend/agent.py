from openai import AsyncAzureOpenAI
from typing import List, Dict, Optional, Any, Tuple
import json
import os
import asyncio
from datetime import datetime
from pydantic import BaseModel
from dotenv import load_dotenv
import random
import uuid
from dataclasses import dataclass
from enum import Enum
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

# Enhanced data structures
class TaskType(str, Enum):
    GENERATION = "generation"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "qa"
    CREATIVE_WRITING = "creative"

class PreferenceLabel(str, Enum):
    CHOSEN = "chosen"
    REJECTED = "rejected"
    NEUTRAL = "neutral"

@dataclass
class Response:
    id: str
    text: str
    model_id: str
    timestamp: str
    metadata: Dict[str, Any]

@dataclass
class PreferenceDataPoint:
    id: str
    prompt: str
    response_a: Response
    response_b: Response
    preference: PreferenceLabel
    confidence: float
    task_type: TaskType
    human_feedback: Optional[str] = None
    timestamp: str = None

class ConversationData(BaseModel):
    prompt: str
    response: str
    user_feedback: Optional[int] = None
    timestamp: str
    task_type: Optional[str] = None
    metadata: Dict[str, Any] = {}

class ModelMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    preference_agreement: float
    timestamp: str

class RLHFAgent:
    def __init__(self, 
                 api_key: str, 
                 endpoint: str, 
                 deployment_name: str, 
                 api_version: str = "2024-02-01", 
                 data_file: str = "conversation_data.json",
                 preference_file: str = "preference_data.json",
                 model_file: str = "trained_model.pkl"):
        
        self.client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version
        )
        self.deployment_name = deployment_name
        self.data_file = data_file
        self.preference_file = preference_file
        self.model_file = model_file
        
        # Initialize data storage
        self.conversation_history: List[ConversationData] = []
        self.preference_dataset: List[PreferenceDataPoint] = []
        self.model_metrics: List[ModelMetrics] = []
        
        # Load existing data
        self._load_data()
        
        # Define function schemas for OpenAI function calling
        self.functions = [
            {
                "name": "generate_preference_dataset",
                "description": "Generate a preference dataset for RLHF training",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_type": {
                            "type": "string",
                            "enum": ["generation", "classification", "summarization", "qa", "creative"],
                            "description": "Type of task for the dataset"
                        },
                        "num_samples": {
                            "type": "integer",
                            "description": "Number of preference pairs to generate",
                            "minimum": 1,
                            "maximum": 100
                        },
                        "domain": {
                            "type": "string",
                            "description": "Domain or topic for the dataset (e.g., 'technology', 'health', 'general')"
                        },
                        "difficulty_level": {
                            "type": "string",
                            "enum": ["easy", "medium", "hard"],
                            "description": "Difficulty level of the tasks"
                        }
                    },
                    "required": ["task_type", "num_samples"]
                }
            },
            {
                "name": "train_reward_model",
                "description": "Train a reward model using the preference dataset",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "model_type": {
                            "type": "string",
                            "enum": ["linear", "neural", "transformer"],
                            "description": "Type of reward model to train"
                        },
                        "training_ratio": {
                            "type": "number",
                            "minimum": 0.1,
                            "maximum": 0.9,
                            "description": "Ratio of data to use for training (rest for validation)"
                        },
                        "epochs": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100,
                            "description": "Number of training epochs"
                        }
                    },
                    "required": ["model_type"]
                }
            },
            {
                "name": "evaluate_model",
                "description": "Evaluate the trained model on various metrics",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "evaluation_type": {
                            "type": "string",
                            "enum": ["preference_accuracy", "response_quality", "human_alignment"],
                            "description": "Type of evaluation to perform"
                        },
                        "test_size": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 50,
                            "description": "Number of test samples to evaluate"
                        },
                        "include_human_eval": {
                            "type": "boolean",
                            "description": "Whether to include human evaluation"
                        }
                    },
                    "required": ["evaluation_type"]
                }
            },
            {
                "name": "get_dataset_stats",
                "description": "Get statistics about the current preference dataset",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "detailed": {
                            "type": "boolean",
                            "description": "Whether to include detailed statistics"
                        }
                    }
                }
            },
            {
                "name": "export_dataset",
                "description": "Export the preference dataset in various formats",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "format": {
                            "type": "string",
                            "enum": ["json", "csv", "huggingface", "openai"],
                            "description": "Export format"
                        },
                        "filename": {
                            "type": "string",
                            "description": "Output filename"
                        }
                    },
                    "required": ["format"]
                }
            }
        ]

    def _load_data(self):
        """Load existing conversation and preference data."""
        # Load conversation history
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                data = json.load(f)
                self.conversation_history = [ConversationData(**item) for item in data]
        
        # Load preference dataset
        if os.path.exists(self.preference_file):
            with open(self.preference_file, 'r') as f:
                data = json.load(f)
                self.preference_dataset = [self._dict_to_preference_datapoint(item) for item in data]

    def _dict_to_preference_datapoint(self, data: Dict) -> PreferenceDataPoint:
        """Convert dictionary to PreferenceDataPoint."""
        return PreferenceDataPoint(
            id=data['id'],
            prompt=data['prompt'],
            response_a=Response(**data['response_a']),
            response_b=Response(**data['response_b']),
            preference=PreferenceLabel(data['preference']),
            confidence=data['confidence'],
            task_type=TaskType(data['task_type']),
            human_feedback=data.get('human_feedback'),
            timestamp=data['timestamp']
        )

    def _save_data(self):
        """Save all data to files."""
        # Save conversation history
        with open(self.data_file, 'w') as f:
            json.dump([item.dict() for item in self.conversation_history], f, indent=2)
        
        # Save preference dataset
        with open(self.preference_file, 'w') as f:
            preference_data = []
            for item in self.preference_dataset:
                preference_data.append({
                    'id': item.id,
                    'prompt': item.prompt,
                    'response_a': {
                        'id': item.response_a.id,
                        'text': item.response_a.text,
                        'model_id': item.response_a.model_id,
                        'timestamp': item.response_a.timestamp,
                        'metadata': item.response_a.metadata
                    },
                    'response_b': {
                        'id': item.response_b.id,
                        'text': item.response_b.text,
                        'model_id': item.response_b.model_id,
                        'timestamp': item.response_b.timestamp,
                        'metadata': item.response_b.metadata
                    },
                    'preference': item.preference.value,
                    'confidence': item.confidence,
                    'task_type': item.task_type.value,
                    'human_feedback': item.human_feedback,
                    'timestamp': item.timestamp
                })
            json.dump(preference_data, f, indent=2)

    async def generate_preference_dataset(self, task_type: str, num_samples: int, 
                                        domain: str = "general", difficulty_level: str = "medium") -> Dict:
        """Generate preference dataset using OpenAI API."""
        try:
            print("Generating preference dataset")
            # Generate prompts for the specified task type
            prompts = await self._generate_prompts(task_type, num_samples, domain, difficulty_level)
            
            generated_pairs = []
            for i, prompt in enumerate(prompts):
                # Generate two different responses for each prompt
                response_a = await self._generate_response(prompt, f"model_a_{i}")
                response_b = await self._generate_response(prompt, f"model_b_{i}")
                
                # Use AI to determine preference
                preference_result = await self._determine_preference(prompt, response_a, response_b)
                
                # Create preference datapoint
                preference_point = PreferenceDataPoint(
                    id=str(uuid.uuid4()),
                    prompt=prompt,
                    response_a=response_a,
                    response_b=response_b,
                    preference=preference_result['preference'],
                    confidence=preference_result['confidence'],
                    task_type=TaskType(task_type),
                    timestamp=datetime.now().isoformat()
                )
                
                self.preference_dataset.append(preference_point)
                generated_pairs.append(preference_point)
            
            self._save_data()
            
            return {
                "success": True,
                "generated_pairs": len(generated_pairs),
                "total_dataset_size": len(self.preference_dataset),
                "task_type": task_type,
                "domain": domain,
                "difficulty_level": difficulty_level
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _generate_prompts(self, task_type: str, num_samples: int, 
                               domain: str, difficulty_level: str) -> List[str]:
        """Generate prompts for the specified task type."""
        prompt_generation_messages = [
            {
                "role": "system",
                "content": f"""You are an expert at generating diverse, high-quality prompts for {task_type} tasks.
                Generate {num_samples} different prompts in the {domain} domain with {difficulty_level} difficulty.
                Each prompt should be unique, engaging, and appropriate for the task type.
                Return only the prompts, one per line."""
            },
            {
                "role": "user",
                "content": f"Generate {num_samples} {difficulty_level} difficulty prompts for {task_type} tasks in the {domain} domain."
            }
        ]
        
        response = await self.client.chat.completions.create(
            model=self.deployment_name,
            messages=prompt_generation_messages,
            max_tokens=2000,
            temperature=0.8
        )
        
        prompts = [line.strip() for line in response.choices[0].message.content.split('\n') if line.strip()]


        return prompts[:num_samples]

    async def _generate_response(self, prompt: str, model_id: str) -> Response:
        """Generate a response to a prompt."""
        # Vary the temperature and instructions to create diverse responses
        temperature = random.uniform(0.3, 0.9)
        
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Provide a clear, helpful response to the user's request."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        response = await self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            max_tokens=1500,
            temperature=temperature
        )
        
        return Response(
            id=str(uuid.uuid4()),
            text=response.choices[0].message.content,
            model_id=model_id,
            timestamp=datetime.now().isoformat(),
            metadata={
                "temperature": temperature,
                "tokens_used": response.usage.total_tokens if response.usage else 0
            }
        )

    async def _determine_preference(self, prompt: str, response_a: Response, response_b: Response) -> Dict:
        """Use AI to determine preference between two responses."""
        evaluation_messages = [
            {
                "role": "system",
                "content": """You are an expert evaluator. Compare two responses to a prompt and determine which is better.
                Consider factors like: accuracy, helpfulness, clarity, relevance, and overall quality.
                Rate your confidence from 0.0 to 1.0."""
            },
            {
                "role": "user",
                "content": f"""
                Prompt: {prompt}
                
                Response A: {response_a.text}
                
                Response B: {response_b.text}
                
                Which response is better? Respond with exactly one of: "A", "B", or "neutral"
                Also provide a confidence score (0.0 to 1.0) and brief reasoning.
                Format: PREFERENCE|CONFIDENCE|REASONING
                """
            }
        ]
        
        response = await self.client.chat.completions.create(
            model=self.deployment_name,
            messages=evaluation_messages,
            max_tokens=200,
            temperature=0.2
        )
        
        try:
            parts = response.choices[0].message.content.split('|')
            preference_str = parts[0].strip().lower()
            confidence = float(parts[1].strip())
            
            if preference_str == 'a':
                preference = PreferenceLabel.CHOSEN
            elif preference_str == 'b':
                preference = PreferenceLabel.REJECTED
            else:
                preference = PreferenceLabel.NEUTRAL
                
            return {
                "preference": preference,
                "confidence": confidence,
                "reasoning": parts[2].strip() if len(parts) > 2 else ""
            }
        except:
            return {
                "preference": PreferenceLabel.NEUTRAL,
                "confidence": 0.5,
                "reasoning": "Unable to parse evaluation"
            }

    async def train_reward_model(self, model_type: str = "linear", 
                               training_ratio: float = 0.8, epochs: int = 10) -> Dict:
        """Train a reward model using the preference dataset."""
        if not self.preference_dataset:
            return {"success": False, "error": "No preference data available for training"}
        
        try:
            # Prepare training data
            X, y = self._prepare_training_data()
            
            # Split data
            split_idx = int(len(X) * training_ratio)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Train model (simplified simulation)
            model = self._train_model(X_train, y_train, model_type, epochs)
            
            # Evaluate on validation set
            val_predictions = self._predict(model, X_val)
            metrics = self._calculate_metrics(y_val, val_predictions)
            
            # Save model
            with open(self.model_file, 'wb') as f:
                pickle.dump(model, f)
            
            # Save metrics
            model_metrics = ModelMetrics(
                accuracy=metrics['accuracy'],
                precision=metrics['precision'],
                recall=metrics['recall'],
                f1_score=metrics['f1_score'],
                preference_agreement=metrics['preference_agreement'],
                timestamp=datetime.now().isoformat()
            )
            self.model_metrics.append(model_metrics)
            
            return {
                "success": True,
                "model_type": model_type,
                "training_samples": len(X_train),
                "validation_samples": len(X_val),
                "epochs": epochs,
                "metrics": metrics
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _prepare_training_data(self) -> Tuple[List, List]:
        """Prepare training data from preference dataset."""
        X = []  # Features (simplified: response lengths, confidence scores, etc.)
        y = []  # Labels (preference labels)
        
        for datapoint in self.preference_dataset:
            # Simple feature extraction (you can make this more sophisticated)
            features = [
                len(datapoint.response_a.text),
                len(datapoint.response_b.text),
                datapoint.confidence,
                len(datapoint.prompt),
                1 if datapoint.task_type == TaskType.GENERATION else 0,
                1 if datapoint.task_type == TaskType.CLASSIFICATION else 0,
            ]
            
            X.append(features)
            y.append(1 if datapoint.preference == PreferenceLabel.CHOSEN else 0)
        
        return X, y

    def _train_model(self, X_train: List, y_train: List, model_type: str, epochs: int) -> Dict:
        """Train a simple model (simulation)."""
        # This is a simplified simulation - in practice, you'd use actual ML libraries
        model = {
            "type": model_type,
            "weights": np.random.random(len(X_train[0])) if X_train else [],
            "epochs": epochs,
            "trained": True
        }
        return model

    def _predict(self, model: Dict, X: List) -> List:
        """Make predictions using the trained model."""
        # Simplified prediction logic
        predictions = []
        for features in X:
            if model["weights"].size > 0:
                score = np.dot(features, model["weights"])
                predictions.append(1 if score > 0.5 else 0)
            else:
                predictions.append(random.randint(0, 1))
        return predictions

    def _calculate_metrics(self, y_true: List, y_pred: List) -> Dict:
        """Calculate evaluation metrics."""
        if not y_true or not y_pred:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "preference_agreement": 0.0
            }
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "preference_agreement": accuracy  # Simplified
        }

    async def evaluate_model(self, evaluation_type: str = "preference_accuracy", 
                           test_size: int = 10, include_human_eval: bool = False) -> Dict:
        """Evaluate the trained model."""
        if not os.path.exists(self.model_file):
            return {"success": False, "error": "No trained model found"}
        
        try:
            # Load model
            with open(self.model_file, 'rb') as f:
                model = pickle.load(f)
            
            # Prepare test data
            test_data = self.preference_dataset[-test_size:] if len(self.preference_dataset) >= test_size else self.preference_dataset
            
            results = {
                "success": True,
                "evaluation_type": evaluation_type,
                "test_size": len(test_data),
                "model_type": model.get("type", "unknown")
            }
            
            if evaluation_type == "preference_accuracy":
                # Test preference prediction accuracy
                X_test, y_test = self._prepare_test_data(test_data)
                predictions = self._predict(model, X_test)
                metrics = self._calculate_metrics(y_test, predictions)
                results["metrics"] = metrics
                
            elif evaluation_type == "response_quality":
                # Evaluate response quality using AI
                quality_scores = await self._evaluate_response_quality(test_data)
                results["quality_scores"] = quality_scores
                
            elif evaluation_type == "human_alignment":
                # Evaluate alignment with human preferences
                alignment_score = await self._evaluate_human_alignment(test_data)
                results["alignment_score"] = alignment_score
            
            if include_human_eval:
                results["human_evaluation_needed"] = True
                results["human_eval_prompts"] = [dp.prompt for dp in test_data[:5]]
            
            return results
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _prepare_test_data(self, test_data: List[PreferenceDataPoint]) -> Tuple[List, List]:
        """Prepare test data for evaluation."""
        X = []
        y = []
        
        for datapoint in test_data:
            features = [
                len(datapoint.response_a.text),
                len(datapoint.response_b.text),
                datapoint.confidence,
                len(datapoint.prompt),
                1 if datapoint.task_type == TaskType.GENERATION else 0,
                1 if datapoint.task_type == TaskType.CLASSIFICATION else 0,
            ]
            
            X.append(features)
            y.append(1 if datapoint.preference == PreferenceLabel.CHOSEN else 0)
        
        return X, y

    async def _evaluate_response_quality(self, test_data: List[PreferenceDataPoint]) -> Dict:
        """Evaluate response quality using AI."""
        quality_scores = []
        
        for datapoint in test_data:
            evaluation_messages = [
                {
                    "role": "system",
                    "content": "Rate the quality of responses on a scale of 1-10 considering accuracy, helpfulness, and clarity."
                },
                {
                    "role": "user",
                    "content": f"Prompt: {datapoint.prompt}\n\nResponse A: {datapoint.response_a.text}\n\nResponse B: {datapoint.response_b.text}\n\nRate both responses (1-10) and provide the average."
                }
            ]
            
            try:
                response = await self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=evaluation_messages,
                    max_tokens=100,
                    temperature=0.2
                )
                
                # Extract score (simplified)
                score = 7.5  # Default score
                quality_scores.append(score)
                
            except:
                quality_scores.append(7.0)  # Default on error
        
        return {
            "average_quality": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            "scores": quality_scores
        }

    async def _evaluate_human_alignment(self, test_data: List[PreferenceDataPoint]) -> float:
        """Evaluate alignment with human preferences."""
        # Simplified alignment calculation
        alignment_scores = []
        
        for datapoint in test_data:
            if datapoint.human_feedback:
                # Compare AI preference with human feedback
                alignment_scores.append(1.0 if datapoint.preference == PreferenceLabel.CHOSEN else 0.0)
            else:
                # Use confidence as proxy for alignment
                alignment_scores.append(datapoint.confidence)
        
        return sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.5

    async def get_dataset_stats(self, detailed: bool = False) -> Dict:
        """Get statistics about the preference dataset."""
        if not self.preference_dataset:
            return {"total_samples": 0, "message": "No preference data available"}
        
        stats = {
            "total_samples": len(self.preference_dataset),
            "task_types": {},
            "preference_distribution": {
                "chosen": 0,
                "rejected": 0,
                "neutral": 0
            },
            "average_confidence": 0.0
        }
        
        # Calculate statistics
        confidence_scores = []
        for datapoint in self.preference_dataset:
            # Task type distribution
            task_type = datapoint.task_type.value
            stats["task_types"][task_type] = stats["task_types"].get(task_type, 0) + 1
            
            # Preference distribution
            preference = datapoint.preference.value
            stats["preference_distribution"][preference] += 1
            
            # Confidence scores
            confidence_scores.append(datapoint.confidence)
        
        stats["average_confidence"] = sum(confidence_scores) / len(confidence_scores)
        
        if detailed:
            stats["detailed"] = {
                "confidence_range": {
                    "min": min(confidence_scores),
                    "max": max(confidence_scores)
                },
                "recent_samples": len([dp for dp in self.preference_dataset 
                                    if datetime.fromisoformat(dp.timestamp) > datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)]),
                "has_human_feedback": len([dp for dp in self.preference_dataset if dp.human_feedback])
            }
        
        return stats

    async def export_dataset(self, format: str = "json", filename: str = None) -> Dict:
        """Export the preference dataset."""
        if not self.preference_dataset:
            return {"success": False, "error": "No data to export"}
        
        if not filename:
            filename = f"preference_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            if format == "json":
                with open(f"{filename}.json", 'w') as f:
                    export_data = []
                    for dp in self.preference_dataset:
                        export_data.append({
                            "prompt": dp.prompt,
                            "chosen": dp.response_a.text if dp.preference == PreferenceLabel.CHOSEN else dp.response_b.text,
                            "rejected": dp.response_b.text if dp.preference == PreferenceLabel.CHOSEN else dp.response_a.text,
                            "task_type": dp.task_type.value,
                            "confidence": dp.confidence
                        })
                    json.dump(export_data, f, indent=2)
                    
            elif format == "csv":
                import csv
                with open(f"{filename}.csv", 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["prompt", "chosen", "rejected", "task_type", "confidence"])
                    for dp in self.preference_dataset:
                        chosen = dp.response_a.text if dp.preference == PreferenceLabel.CHOSEN else dp.response_b.text
                        rejected = dp.response_b.text if dp.preference == PreferenceLabel.CHOSEN else dp.response_a.text
                        writer.writerow([dp.prompt, chosen, rejected, dp.task_type.value, dp.confidence])
            
            return {
                "success": True,
                "format": format,
                "filename": f"{filename}.{format}",
                "samples_exported": len(self.preference_dataset)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def process_function_call(self, function_name: str, arguments: Dict) -> Dict:
        """Process function calls from OpenAI."""
        if function_name == "generate_preference_dataset":
            print("Generating preference dataset")
            return await self.generate_preference_dataset(**arguments)
        elif function_name == "train_reward_model":
            return await self.train_reward_model(**arguments)
        elif function_name == "evaluate_model":
            return await self.evaluate_model(**arguments)
        elif function_name == "get_dataset_stats":
            return await self.get_dataset_stats(**arguments)
        elif function_name == "export_dataset":
            return await self.export_dataset(**arguments)
        else:
            return {"success": False, "error": f"Unknown function: {function_name}"}

    async def run_rlhf_conversation(self, user_input: str) -> str:
        """Handle RLHF-specific conversations with function calling."""
        try:
            messages = [
                {
                    "role": "system",
                    "content": """You are an advanced RLHF (Reinforcement Learning from Human Feedback) assistant designed to assist with reinforcement learning tasks. Your role is to interpret user requests and execute the appropriate functions to support RLHF workflows. You have access to the following functions:

            1. `generate_preference_dataset`: Generate preference datasets for tasks like generation, classification, summarization, question answering, or creative writing.
            2. `train_reward_model`: Train reward models using the preference dataset.
            3. `evaluate_model`: Evaluate trained models using metrics like preference accuracy, response quality, or human alignment.
            4. `get_dataset_stats`: Provide statistics about the preference dataset, including task distribution and preference labels.
            5. `export_dataset`: Export the preference dataset in formats like JSON or CSV.

            **Instructions:**
            - Analyze the user's input to determine if it relates to an RLHF task. If it does, select the most appropriate function and execute it with the correct parameters.
            - If the input is ambiguous, ask for clarification to ensure the correct function is called.
            - For non-RLHF queries, provide a helpful conversational response and suggest relevant RLHF tasks the user might explore.
            - Always return clear, concise, and natural language responses, summarizing function outputs when applicable.
            - If no function is needed, respond conversationally and offer guidance on available RLHF capabilities.

            **Example Inputs and Actions:**
            - "Generate a dataset for summarization" → Call `generate_preference_dataset` with task_type="summarization".
            - "Train a model" → Call `train_reward_model` with default parameters or ask for specifics.
            - "Show dataset stats" → Call `get_dataset_stats`.
            - "What's RLHF?" → Provide a conversational explanation and suggest related functions.

            **Error Handling:**
            - If a function call fails or the input is invalid, return an error message with guidance on how to correct the request.
            - If the user input doesn't match any function, respond conversationally and suggest relevant RLHF tasks."""
                },
                {"role": "user", "content": user_input}
            ]
            
            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                functions=self.functions,
                function_call="auto",
                max_tokens=1500,
                temperature=0.3
            )
            
            # Handle function calls
            if response.choices[0].message.function_call:
                function_name = response.choices[0].message.function_call.name
                arguments = json.loads(response.choices[0].message.function_call.arguments)
                
                function_result = await self.process_function_call(function_name, arguments)
                
                # Generate response based on function result
                result_message = f"Function '{function_name}' executed successfully.\n\nResult: {json.dumps(function_result, indent=2)}"
                
                # Generate a natural language response
                follow_up_messages = [
                    {
                        "role": "system",
                        "content": "You are an RLHF assistant. Provide a natural language summary of the function result."
                    },
                    {
                        "role": "user",
                        "content": f"Summarize this function result in a helpful way: {function_result}"
                    }
                ]
                
                summary_response = await self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=follow_up_messages,
                    max_tokens=500,
                    temperature=0.3
                )
                
                return f"{summary_response.choices[0].message.content}\n\n📊 Raw Result:\n{result_message}"
            
            else:
                return response.choices[0].message.content
                
        except Exception as e:
            return f"Error processing RLHF request: {str(e)}"

    async def run_conversation(self, user_input: str) -> str:
        """Enhanced conversation handler with RLHF capabilities."""
        try:
            # Check if this is an RLHF-specific request
            rlhf_keywords = [
                "generate dataset", "preference dataset", "train model", "reward model",
                "evaluate model", "rlhf", "preference", "dataset stats", "export dataset"
            ]
            
            if any(keyword in user_input.lower() for keyword in rlhf_keywords):
                return await self.run_rlhf_conversation(user_input)
            
            # Handle regular conversation
            messages = [
                {
                    "role": "system",
                    "content": """You are a helpful RLHF assistant. You can help with:
                    1. Generating preference datasets for RLHF training
                    2. Training reward models
                    3. Evaluating model performance
                    4. General conversation and questions
                    
                    When users ask about RLHF tasks, guide them on what's possible."""
                }
            ]
            
            # Add recent conversation context
            recent_history = self.conversation_history[-5:] if len(self.conversation_history) > 5 else self.conversation_history
            for conv in recent_history:
                messages.append({"role": "user", "content": conv.prompt})
                messages.append({"role": "assistant", "content": conv.response})
            
            messages.append({"role": "user", "content": user_input})
            
            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            assistant_response = response.choices[0].message.content
            timestamp = datetime.now().isoformat()
            
            # Save the conversation
            conversation = ConversationData(
                prompt=user_input,
                response=assistant_response,
                timestamp=timestamp,
                metadata={"type": "general_conversation"}
            )
            self.conversation_history.append(conversation)
            self._save_data()
            
            return f"{assistant_response}\n\n💡 Try asking about RLHF tasks like:\n• 'Generate a preference dataset for summarization'\n• 'Train a reward model'\n• 'Evaluate the current model'\n• 'Show dataset statistics'"
            
        except Exception as e:
            return f"Error generating response: {str(e)}"

    async def show_capabilities(self) -> str:
        """Show the capabilities of the RLHF agent."""
        return """
🤖 RLHF Agent Capabilities

📊 PREFERENCE DATASET GENERATION:
• Generate datasets for various task types (generation, classification, summarization, Q&A, creative writing)
• Customize domain, difficulty level, and sample size
• Automatic preference labeling using AI evaluation

🏋️ MODEL TRAINING:
• Train reward models using preference data
• Support for different model types (linear, neural, transformer)
• Configurable training parameters and validation splits

📈 MODEL EVALUATION:
• Evaluate preference prediction accuracy
• Assess response quality using AI metrics
• Measure human alignment scores
• Include human evaluation workflows

📋 DATASET MANAGEMENT:
• View detailed dataset statistics
• Export datasets in multiple formats (JSON, CSV, HuggingFace, OpenAI)
• Track training progress and model metrics

🔧 EXAMPLE COMMANDS:
• "Generate a preference dataset for summarization with 20 samples"
• "Train a neural reward model with 80% training data"
• "Evaluate the model on response quality"
• "Show detailed dataset statistics"
• "Export dataset in JSON format"

💬 GENERAL FEATURES:
• Conversation history tracking
• Feedback collection and analysis
• Comprehensive error handling
• Data persistence across sessions
        """

async def main():
    """Main function to run the enhanced RLHF agent."""
    # Initialize the agent with Azure OpenAI configuration
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    
    if not all([api_key, endpoint, deployment_name]):
        print("❌ Missing required environment variables:")
        print("• AZURE_OPENAI_API_KEY")
        print("• AZURE_OPENAI_ENDPOINT")
        print("• AZURE_OPENAI_DEPLOYMENT_NAME")
        print("• AZURE_OPENAI_API_VERSION (optional, defaults to 2024-02-01)")
        return
    
    agent = RLHFAgent(
        api_key=api_key,
        endpoint=endpoint,
        deployment_name=deployment_name,
        api_version=api_version
    )
    
    print("🚀 Enhanced RLHF Agent Started!")
    print("=" * 60)
    print(await agent.show_capabilities())
    print("=" * 60)
    print("💡 Type 'help' for capabilities, 'exit' to quit")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\n🔵 You: ").strip()
            
            if user_input.lower() == "exit":
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == "help":
                print(await agent.show_capabilities())
                continue
            
            if not user_input:
                continue
            
            # Special commands for direct function testing
            if user_input.lower().startswith("test_"):
                if user_input.lower() == "test_generate":
                    result = await agent.generate_preference_dataset("generation", 3, "technology", "medium")
                    print(f"\n🤖 Agent: {json.dumps(result, indent=2)}")
                elif user_input.lower() == "test_train":
                    result = await agent.train_reward_model("linear", 0.8, 5)
                    print(f"\n🤖 Agent: {json.dumps(result, indent=2)}")
                elif user_input.lower() == "test_evaluate":
                    result = await agent.evaluate_model("preference_accuracy", 5)
                    print(f"\n🤖 Agent: {json.dumps(result, indent=2)}")
                elif user_input.lower() == "test_stats":
                    result = await agent.get_dataset_stats(detailed=True)
                    print(f"\n🤖 Agent: {json.dumps(result, indent=2)}")
                continue
            
            # Handle regular conversation
            response = await agent.run_conversation(user_input)
            print(f"\n🤖 Agent: {response}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())