"""
Conversational Ollama Service for Interactive Pest Management Consultation
Maintains chat history and provides context-aware responses
"""

import ollama
import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys
from datetime import datetime

# Import configuration
sys.path.append(str(Path(__file__).parent.parent))
from config.config import HARMFUL_PEST_CLASSES, ECONOMIC_IMPACT, TREATMENT_URGENCY, OLLAMA_CONFIG
from presentation.ollama_service import OllamaService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationalOllamaService(OllamaService):
    """
    Enhanced Ollama service with conversational capabilities and chat history
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize conversational Ollama service
        """
        super().__init__(model_name)
        self.chat_history = []
        self.current_pest_context = None
        self.session_start = datetime.now()
        
    def start_new_session(self, pest_class: str = None, confidence: float = None):
        """
        Start a new conversation session with optional pest context
        
        Args:
            pest_class: Currently identified pest
            confidence: Model confidence for the identification
        """
        self.chat_history = []
        self.current_pest_context = {
            'pest_class': pest_class,
            'confidence': confidence,
            'identified_at': datetime.now()
        } if pest_class else None
        self.session_start = datetime.now()
        
        logger.info(f"🆕 Started new conversation session for {pest_class if pest_class else 'general consultation'}")
    
    def add_pest_context(self, pest_class: str, confidence: float):
        """
        Add or update pest identification context to current conversation
        
        Args:
            pest_class: Identified pest class
            confidence: Model confidence score
        """
        self.current_pest_context = {
            'pest_class': pest_class,
            'confidence': confidence,
            'identified_at': datetime.now()
        }
        
        # Add system message about new pest identification
        context_message = f"🔍 **New Pest Identified:** {pest_class.title()} (Confidence: {confidence:.1%})"
        self.chat_history.append({
            'role': 'system',
            'content': context_message,
            'timestamp': datetime.now()
        })
        
        logger.info(f"🎯 Added pest context: {pest_class} ({confidence:.1%})")
    
    def chat(self, user_message: str) -> str:
        """
        Process a conversational message from the user
        
        Args:
            user_message: User's question or message
            
        Returns:
            str: AI response
        """
        if not self.is_available:
            return self._fallback_chat_response(user_message)
        
        try:
            # Add user message to history
            self.chat_history.append({
                'role': 'user',
                'content': user_message,
                'timestamp': datetime.now()
            })
            
            # Prepare messages for Ollama
            messages = self._prepare_chat_messages()
            
            # Generate response
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options=OLLAMA_CONFIG['generation_params']
            )
            
            ai_response = response['message']['content']
            
            # Add AI response to history
            self.chat_history.append({
                'role': 'assistant',
                'content': ai_response,
                'timestamp': datetime.now()
            })
            
            return self._format_chat_response(ai_response)
            
        except Exception as e:
            logger.error(f"❌ Error in conversational chat: {e}")
            return self._fallback_chat_response(user_message)
    
    def _prepare_chat_messages(self) -> List[Dict]:
        """
        Prepare messages for Ollama chat including system prompt and history
        
        Returns:
            List[Dict]: Formatted messages for Ollama
        """
        messages = [
            {
                "role": "system",
                "content": self._get_conversational_system_prompt()
            }
        ]
        
        # Add pest context if available
        if self.current_pest_context:
            context_msg = self._format_pest_context()
            messages.append({
                "role": "system",
                "content": context_msg
            })
        
        # Add recent chat history (last 10 messages to avoid token limits)
        recent_history = self.chat_history[-10:] if len(self.chat_history) > 10 else self.chat_history
        
        for msg in recent_history:
            if msg['role'] in ['user', 'assistant']:
                messages.append({
                    "role": msg['role'],
                    "content": msg['content']
                })
        
        return messages
    
    def _get_conversational_system_prompt(self) -> str:
        """
        Get enhanced system prompt for conversational interactions
        
        Returns:
            str: Conversational system prompt
        """
        return """You are Dr. GreenThumb, a friendly and knowledgeable organic farming consultant specializing in integrated pest management (IPM). You're having a conversation with a farmer who needs help with pest management.

PERSONALITY & APPROACH:
- Be conversational, helpful, and encouraging
- Use a warm, professional tone like talking to a neighbor
- Ask clarifying questions when needed
- Provide practical, actionable advice
- Show empathy for farming challenges

EXPERTISE FOCUS:
- Organic and sustainable pest management only
- Integrated Pest Management (IPM) principles
- Biological controls and natural methods
- Prevention strategies and monitoring
- Economic considerations for small farmers
- Environmental safety and sustainability

CONVERSATION GUIDELINES:
- Remember previous parts of our conversation
- Build on what we've already discussed
- If a pest was identified, reference it in context
- Provide specific, step-by-step guidance
- Suggest follow-up questions or next steps
- Acknowledge when you need more information

RESPONSE FORMAT:
- Use clear, conversational language
- Include practical tips and timing
- Mention costs when relevant
- Suggest monitoring and follow-up
- Use simple formatting (bullets, numbers)
- Keep responses focused but thorough

SAFETY REMINDERS:
- Always prioritize human and environmental safety
- Emphasize proper application methods
- Mention protective equipment when needed
- Suggest consulting local experts for complex issues

Remember: You're here to help farmers succeed with organic methods while protecting their crops, health, and environment."""
    
    def _format_pest_context(self) -> str:
        """
        Format current pest context for the conversation
        
        Returns:
            str: Formatted pest context
        """
        if not self.current_pest_context:
            return ""
        
        pest = self.current_pest_context['pest_class']
        confidence = self.current_pest_context['confidence']
        
        # Get pest characteristics
        economic_impact = ECONOMIC_IMPACT.get(pest, 3)
        urgency = TREATMENT_URGENCY.get(pest, 'Medium')
        
        context = f"""CURRENT PEST IDENTIFICATION CONTEXT:
- Identified Pest: {pest.title()}
- Confidence Level: {confidence:.1%}
- Economic Impact: {economic_impact}/5
- Treatment Urgency: {urgency}
- Identified: {self.current_pest_context['identified_at'].strftime('%Y-%m-%d %H:%M')}

The farmer has uploaded an image and this pest was identified. They may ask follow-up questions about treatment, prevention, timing, costs, or alternatives. Reference this pest identification in your responses when relevant."""
        
        return context
    
    def _format_chat_response(self, response: str) -> str:
        """
        Format the AI chat response with conversational elements
        
        Args:
            response: Raw AI response
            
        Returns:
            str: Formatted conversational response
        """
        # Add conversational header if this is the first response after pest identification
        if (self.current_pest_context and 
            len([msg for msg in self.chat_history if msg['role'] == 'user']) == 1):
            
            pest = self.current_pest_context['pest_class']
            confidence = self.current_pest_context['confidence']
            
            header = f"🌱 **Dr. GreenThumb here!** I see you've identified a **{pest.title()}** with {confidence:.1%} confidence. Let me help you with that!\n\n"
            return header + response
        
        return response
    
    def _fallback_chat_response(self, user_message: str) -> str:
        """
        Provide fallback responses when Ollama is unavailable
        
        Args:
            user_message: User's message
            
        Returns:
            str: Fallback response
        """
        # Simple keyword-based responses
        message_lower = user_message.lower()
        
        if any(word in message_lower for word in ['cost', 'price', 'expensive', 'cheap']):
            return """💰 **Cost-Effective Organic Solutions:**

Most organic treatments are quite affordable:
• Neem oil: $10-15 for season-long protection
• Diatomaceous earth: $15-20 for multiple applications  
• Beneficial insects: $20-30 one-time investment
• Companion plants: Seeds cost $2-5 per packet

**Money-saving tips:**
• Make your own soap sprays (dish soap + water)
• Encourage natural predators (free!)
• Practice prevention (saves money long-term)

⚠️ Note: AI chat unavailable. For detailed cost analysis, ensure Ollama is running."""

        elif any(word in message_lower for word in ['when', 'timing', 'time', 'apply']):
            return """⏰ **Treatment Timing Guidelines:**

**General Rules:**
• Early morning or evening (avoid hot sun)
• Before rain but not during wet periods
• When pests are most active (varies by species)
• During calm weather (no strong winds)

**Seasonal Considerations:**
• Spring: Focus on prevention and monitoring
• Summer: Active treatment during peak pest season
• Fall: Clean-up and preparation for next year

⚠️ Note: AI chat unavailable. For specific timing advice, ensure Ollama is running."""

        elif any(word in message_lower for word in ['alternative', 'other', 'different', 'else']):
            return """🔄 **Alternative Organic Approaches:**

If your current treatment isn't working:

1. **Biological Controls:** Introduce beneficial insects
2. **Physical Barriers:** Row covers, copper strips, traps
3. **Cultural Methods:** Crop rotation, companion planting
4. **Natural Sprays:** Soap, oil, or botanical extracts
5. **Environmental Changes:** Improve drainage, spacing, airflow

**Combination Approach:** Often most effective!

⚠️ Note: AI chat unavailable. For personalized alternatives, ensure Ollama is running."""

        else:
            return f"""🤖 **Chat Currently Limited**

I'd love to help you with: "{user_message}"

**Quick Organic Pest Management Tips:**
• Start with the gentlest methods first
• Monitor regularly for early detection
• Encourage beneficial insects
• Keep plants healthy and stress-free
• Use integrated approaches for best results

⚠️ **To enable full conversational AI:** Please ensure Ollama is running with the configured model.

**Need immediate help?** Contact your local agricultural extension office."""
    
    def get_chat_summary(self) -> Dict:
        """
        Get a summary of the current chat session
        
        Returns:
            Dict: Chat session summary
        """
        return {
            'session_duration': str(datetime.now() - self.session_start),
            'message_count': len(self.chat_history),
            'current_pest': self.current_pest_context['pest_class'] if self.current_pest_context else None,
            'pest_confidence': self.current_pest_context['confidence'] if self.current_pest_context else None,
            'last_activity': self.chat_history[-1]['timestamp'] if self.chat_history else self.session_start
        }
    
    def export_chat_history(self) -> str:
        """
        Export chat history as formatted text
        
        Returns:
            str: Formatted chat history
        """
        if not self.chat_history:
            return "No chat history available."
        
        export_text = f"# Pest Management Consultation\n"
        export_text += f"**Session Start:** {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        if self.current_pest_context:
            pest = self.current_pest_context['pest_class']
            confidence = self.current_pest_context['confidence']
            export_text += f"**Identified Pest:** {pest.title()} ({confidence:.1%} confidence)\n\n"
        
        export_text += "## Conversation History\n\n"
        
        for msg in self.chat_history:
            if msg['role'] == 'user':
                export_text += f"**👨‍🌾 Farmer:** {msg['content']}\n\n"
            elif msg['role'] == 'assistant':
                export_text += f"**🌱 Dr. GreenThumb:** {msg['content']}\n\n"
        
        return export_text