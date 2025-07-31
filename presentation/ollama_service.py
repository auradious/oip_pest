"""
Ollama Integration Service for Organic Farm Pest Management AI System
Provides AI-powered treatment recommendations using local LLM models
"""

import ollama
import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys

# Import configuration
sys.path.append(str(Path(__file__).parent.parent))
from config.config import HARMFUL_PEST_CLASSES, ECONOMIC_IMPACT, TREATMENT_URGENCY, OLLAMA_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaService:
    """
    Service class for integrating Ollama LLM for pest treatment recommendations
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize Ollama service
        
        Args:
            model_name: Name of the Ollama model to use (configured in config.py)
        """
        self.model_name = model_name or OLLAMA_CONFIG['default_model']
        self.client = ollama.Client()
        self.is_available = False
        self.check_ollama_availability()
        
    def check_ollama_availability(self) -> bool:
        """
        Check if Ollama is running and the model is available
        
        Returns:
            bool: True if Ollama is available and model is ready
        """
        try:
            # Check if Ollama is running
            models_response = self.client.list()
            
            # Handle different response structures
            if isinstance(models_response, dict) and 'models' in models_response:
                models_list = models_response['models']
            else:
                models_list = models_response
            
            # Extract model names safely
            available_models = []
            for model in models_list:
                if isinstance(model, dict):
                    # Try different possible keys for model name
                    model_name = model.get('name') or model.get('model') or model.get('id', 'unknown')
                    available_models.append(model_name)
                else:
                    available_models.append(str(model))
            
            logger.info(f"📋 Available models: {available_models}")
            
            if self.model_name in available_models:
                self.is_available = True
                logger.info(f"✅ Ollama model '{self.model_name}' is available")
                return True
            else:
                logger.warning(f"⚠️ Model '{self.model_name}' not found. Available models: {available_models}")
                # Try to pull the model
                return self.pull_model()
                
        except Exception as e:
            logger.error(f"❌ Ollama not available: {str(e)}")
            self.is_available = False
            return False
    
    def pull_model(self) -> bool:
        """
        Pull the specified model if not available
        
        Returns:
            bool: True if model was successfully pulled
        """
        try:
            logger.info(f"📥 Pulling model '{self.model_name}'...")
            self.client.pull(self.model_name)
            self.is_available = True
            logger.info(f"✅ Model '{self.model_name}' pulled successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to pull model '{self.model_name}': {e}")
            return False
    
    def get_model_status(self) -> Dict:
        """
        Get the current status of Ollama and the model
        
        Returns:
            Dict: Status information
        """
        try:
            models_response = self.client.list()
            
            # Handle different response structures
            if isinstance(models_response, dict) and 'models' in models_response:
                models_list = models_response['models']
            else:
                models_list = models_response
            
            # Extract model names
            available_models = []
            for model in models_list:
                if isinstance(model, dict):
                    model_name = model.get('name') or model.get('model') or model.get('id', 'unknown')
                    available_models.append(model_name)
                else:
                    available_models.append(str(model))
            
            return {
                "ollama_available": True,
                "model_available": self.model_name in available_models,
                "model_name": self.model_name,
                "available_models": available_models
            }
        except Exception as e:
            return {
                "ollama_available": False,
                "model_available": False,
                "model_name": self.model_name,
                "available_models": [],
                "error": str(e)
            }
    
    def generate_treatment_recommendation(
        self, 
        pest_class: str, 
        confidence: float
    ) -> str:
        """
        Generate comprehensive treatment recommendations using Ollama
        
        Args:
            pest_class: Identified pest class
            confidence: Model confidence score
            
        Returns:
            str: Formatted treatment recommendation
        """
        if not self.is_available:
            return self._fallback_recommendation(pest_class)
        
        try:
            # Prepare context for the LLM
            context = self._prepare_context(pest_class, confidence)
            
            # Generate recommendation using Ollama
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": context
                    }
                ],
                options=OLLAMA_CONFIG['generation_params']
            )
            
            recommendation = response['message']['content']
            return self._format_recommendation(recommendation, pest_class, confidence)
            
        except Exception as e:
            logger.error(f"❌ Error generating recommendation with Ollama: {e}")
            return self._fallback_recommendation(pest_class)
    
    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for the LLM
        
        Returns:
            str: System prompt for organic pest management
        """
        return """You are an expert organic farming consultant specializing in integrated pest management (IPM). 
        Your role is to provide practical, science-based organic treatment recommendations for crop pests.

        Guidelines:
        - Focus ONLY on organic, sustainable, and environmentally-friendly solutions
        - Prioritize biological controls and natural methods
        - Provide specific, actionable steps farmers can implement immediately
        - Include prevention strategies alongside treatment
        - Mention timing considerations (when to apply treatments)
        - Consider economic feasibility for small-scale farmers
        - Always emphasize safety for humans and the environment
        
        IMPORTANT FORMATTING RULES:
        - Use simple, clean text formatting
        - Avoid excessive asterisks, bold text, or complex markdown
        - Use simple bullet points (•) or numbers (1., 2., 3.)
        - Keep sections clearly separated with line breaks
        - Make it easy to read on mobile devices
        
        Format your response with these sections:
        1. Immediate Actions (if urgent)
        2. Organic Treatment Options (ranked by effectiveness)
        3. Prevention Strategies
        4. Monitoring Recommendations
        5. When to Seek Additional Help
        
        Keep responses practical, clean, and concise for field use."""
    
    def _prepare_context(
        self, 
        pest_class: str, 
        confidence: float
    ) -> str:
        """
        Prepare context information for the LLM
        
        Args:
            pest_class: Identified pest class
            confidence: Model confidence score
            
        Returns:
            str: Formatted context for the LLM
        """
        # Get pest characteristics from config
        economic_impact = ECONOMIC_IMPACT.get(pest_class, 3)
        urgency = TREATMENT_URGENCY.get(pest_class, 'Medium')
        
        context = f"""
PEST IDENTIFICATION RESULTS:
- Pest Type: {pest_class.title()}
- Confidence Level: {confidence:.1%}
- Economic Impact Rating: {economic_impact}/5
- Treatment Urgency: {urgency}

Please provide comprehensive organic treatment recommendations for this {pest_class} infestation.
Focus on immediate actions needed and long-term prevention strategies.
Consider the urgency level ({urgency}) and economic impact ({economic_impact}/5) in your recommendations.
"""
        return context
    
    def _format_recommendation(self, recommendation: str, pest_class: str, confidence: float) -> str:
        """
        Format the AI recommendation with additional context
        
        Args:
            recommendation: Raw AI recommendation
            pest_class: Identified pest class
            confidence: Model confidence score
            
        Returns:
            str: Formatted recommendation
        """
        # Get pest characteristics
        economic_impact = ECONOMIC_IMPACT.get(pest_class, 3)
        urgency = TREATMENT_URGENCY.get(pest_class, 'Medium')
        
        # Format header
        urgency_emojis = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
        impact_emojis = {1: "💚", 2: "💛", 3: "🧡", 4: "❤️", 5: "💔"}
        
        header = f"""🤖 **AI-Powered Organic Treatment Plan**
{urgency_emojis[urgency]} **Urgency:** {urgency} | {impact_emojis[economic_impact]} **Economic Impact:** {economic_impact}/5

**Pest Identified:** {pest_class.title()} (Confidence: {confidence:.1%})

"""
        
        return header + recommendation
    
    def _fallback_recommendation(self, pest_class: str) -> str:
        """
        Provide fallback recommendations when Ollama is unavailable
        
        Args:
            pest_class: Identified pest class
            
        Returns:
            str: Basic organic treatment recommendation
        """
        if pest_class not in HARMFUL_PEST_CLASSES:
            return "✅ **No treatment needed** - This appears to be a beneficial insect!"
        
        # Get pest characteristics
        economic_impact = ECONOMIC_IMPACT.get(pest_class, 3)
        urgency = TREATMENT_URGENCY.get(pest_class, 'Medium')
        
        # Basic organic treatments by pest type
        treatments = {
            'beetle': "• Neem oil spray\n• Beneficial nematodes\n• Row covers\n• Hand picking\n• Diatomaceous earth",
            'catterpillar': "• Bacillus thuringiensis (Bt)\n• Row covers\n• Hand picking\n• Beneficial wasps\n• Companion planting with herbs",
            'earwig': "• Beer traps\n• Diatomaceous earth\n• Remove garden debris\n• Beneficial predators\n• Copper strips",
            'grasshopper': "• Row covers\n• Beneficial birds habitat\n• Neem oil\n• Kaolin clay spray\n• Timing of plantings",
            'moth': "• Pheromone traps\n• Bacillus thuringiensis (Bt)\n• Row covers during flight season\n• Beneficial parasitic wasps\n• Light traps",
            'slug': "• Beer traps\n• Copper barriers\n• Diatomaceous earth\n• Iron phosphate baits\n• Remove hiding places",
            'snail': "• Beer traps\n• Copper barriers\n• Diatomaceous earth\n• Hand picking\n• Crushed eggshells",
            'wasp': "• Usually beneficial! Only treat if problematic\n• Remove food sources\n• Seal nest entrances\n• Professional removal if needed",
            'weevil': "• Beneficial nematodes\n• Diatomaceous earth\n• Remove infected plants\n• Crop rotation\n• Sticky traps"
        }
        
        treatment = treatments.get(pest_class, "Consult local agricultural extension for specific recommendations.")
        
        # Format urgency and impact information
        impact_emojis = {1: "💚", 2: "💛", 3: "🧡", 4: "❤️", 5: "💔"}
        urgency_emojis = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
        
        recommendation = f"""**Basic Organic Treatment Plan**
{urgency_emojis[urgency]} **Urgency:** {urgency} | {impact_emojis[economic_impact]} **Economic Impact:** {economic_impact}/5

**Organic Treatments:**
{treatment}

**General Prevention Tips:**
• Regular monitoring and early detection
• Encourage beneficial insects
• Maintain healthy soil and plants
• Use companion planting
• Practice crop rotation

⚠️ **Note:** AI recommendations unavailable. For detailed treatment plans, ensure Ollama is running with the configured model.
"""
        
        return recommendation