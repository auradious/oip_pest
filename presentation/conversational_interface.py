"""
Enhanced Conversational Interface for Organic Farm Pest Management AI System
Includes chat functionality alongside image identification
*CONVERSATIONAL CHATBOT INCLUDED*
"""

import gradio as gr
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.config import GRADIO_CONFIG
from config.languages import LANGUAGES, DEFAULT_LANGUAGE, AVAILABLE_LANGUAGES
from presentation.pest_predictor import PestPredictor
from presentation.conversational_ollama import ConversationalOllamaService

def create_conversational_pest_interface():
    """
    Create enhanced Gradio interface with conversational AI capabilities
    """
    # Initialize services
    predictor = PestPredictor()
    chat_service = ConversationalOllamaService()
    
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="🌱 Conversational Pest Management AI",
        css="""
        .gradio-container {
            max-width: 1200px !important;
            margin: auto !important;
            padding: 20px;
        }
        
        /* Make most text white */
        .gradio-container * {
            color: #ffffff !important;
        }
        
        /* Keep text in white boxes black for readability - STRONG OVERRIDE */
        .gr-textbox, .gr-textbox textarea, .gr-textbox input,
        .gr-textbox *, .gr-textbox textarea *, .gr-textbox input * {
            color: #000000 !important;
            background-color: #ffffff !important;
        }
        
        /* Specific override for identification and treatment textboxes */
        textarea[placeholder*="Upload an image"], 
        textarea[placeholder*="Treatment recommendations"],
        .gr-textbox textarea[readonly],
        .gr-textbox[data-testid] textarea,
        .gr-textbox[data-testid] * {
            color: #000000 !important;
            background-color: #ffffff !important;
        }
        
        /* Make chatbot text WHITE */
        .chatbot, .chatbot *, .chatbot .message, .chatbot .message * {
            color: #ffffff !important;
        }
        
        /* Keep chat input textbox black */
        input[type="text"], textarea {
            color: #000000 !important;
            background-color: #ffffff !important;
        }
        
        /* White text for labels and headers */
        label, .gr-markdown h1, .gr-markdown h2, .gr-markdown h3, .gr-markdown h4 {
            color: #ffffff !important;
        }
        
        /* White text for buttons */
        button, .gr-button {
            color: #ffffff !important;
        }
        
        /* White text for general markdown content */
        .gr-markdown p, .gr-markdown strong, .gr-markdown em {
            color: #ffffff !important;
        }
        
        .chat-container {
            height: 400px;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 10px;
            background: #f9f9f9;
            color: #000000 !important;
        }
        
        .pest-context {
            background: #e8f5e8 !important;
            padding: 10px;
            border-radius: 6px;
            margin-bottom: 10px;
            border-left: 4px solid #4CAF50;
            color: #000000 !important;
        }
        
        /* FORCE BLACK TEXT in pest context with LIGHT BACKGROUND - STRONGEST OVERRIDE */
        .pest-context, .pest-context *, 
        .pest-context p, .pest-context strong, .pest-context em,
        .gradio-container .pest-context,
        .gradio-container .pest-context *,
        .gradio-container .pest-context p,
        .gradio-container .pest-context strong,
        .gradio-container .pest-context em {
            color: #000000 !important;
            background-color: #e8f5e8 !important;
        }
        
        /* Ensure the pest context container itself has light background */
        .gradio-container .pest-context {
            background: #e8f5e8 !important;
            background-color: #e8f5e8 !important;
        }
        
        .chat-message {
            margin-bottom: 10px;
            padding: 8px;
            border-radius: 6px;
            color: #000000 !important;
        }
        
        .user-message {
            background: #e3f2fd;
            margin-left: 20px;
            color: #000000 !important;
        }
        
        .ai-message {
            background: #f1f8e9;
            margin-right: 20px;
            color: #000000 !important;
        }
        
        .section-header {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 15px;
            color: #ffffff !important;
        }
        
        .quick-questions {
            margin-top: 10px;
        }
        
        .quick-questions button {
            margin: 2px;
            font-size: 0.9em;
            color: #ffffff !important;
        }
        
        /* Force black text ONLY for identification and treatment textboxes */
        .gradio-container .gr-textbox textarea[placeholder*="Upload an image"],
        .gradio-container .gr-textbox textarea[placeholder*="Treatment recommendations"],
        .gradio-container .gr-textbox textarea[readonly] {
            color: #000000 !important;
            background-color: #ffffff !important;
        }
        
        /* Keep chat input textbox black for typing */
        .gradio-container .gr-textbox textarea[placeholder*="Ask about treatments"],
        .gradio-container input[type="text"] {
            color: #000000 !important;
            background-color: #ffffff !important;
        }
        
        /* EMERGENCY OVERRIDE - Force black text in light background areas while preserving backgrounds */
        .gradio-container [style*="background: #e8f5e8"],
        .gradio-container [style*="background:#e8f5e8"],
        .gradio-container [style*="background-color: #e8f5e8"],
        .gradio-container [style*="background-color:#e8f5e8"],
        .gradio-container .gr-markdown[class*="pest-context"],
        .gradio-container div[class*="pest-context"] {
            color: #000000 !important;
            background: #e8f5e8 !important;
            background-color: #e8f5e8 !important;
        }
        
        .gradio-container [style*="background: #e8f5e8"] *,
        .gradio-container [style*="background:#e8f5e8"] *,
        .gradio-container [style*="background-color: #e8f5e8"] *,
        .gradio-container [style*="background-color:#e8f5e8"] *,
        .gradio-container .gr-markdown[class*="pest-context"] *,
        .gradio-container div[class*="pest-context"] * {
            color: #000000 !important;
        }
        """
    ) as interface:
        
        # Header
        gr.Markdown("# 🌱 Conversational Pest Management AI")
        gr.Markdown("**Upload an image to identify pests, then chat with Dr. GreenThumb for personalized organic treatment advice!**")
        gr.Markdown("---")
        
        with gr.Row():
            # Left Column: Image Upload & Identification
            with gr.Column(scale=1):
                gr.Markdown("### 📸 Pest Identification", elem_classes="section-header")
                
                image_input = gr.Image(
                    label="Upload Pest Image",
                    type="pil",
                    height=300,
                    sources=["upload"]
                )
                
                identify_btn = gr.Button(
                    "🔍 Identify Pest", 
                    variant="primary",
                    size="lg"
                )
                
                # Identification Results
                identification_output = gr.Textbox(
                    label="Pest Identification Results",
                    lines=4,
                    interactive=False,
                    placeholder="Upload an image and click 'Identify Pest' to see results..."
                )
                
                # Initial Treatment Recommendations
                treatment_output = gr.Textbox(
                    label="Initial Treatment Recommendations",
                    lines=8,
                    interactive=False,
                    placeholder="Treatment recommendations will appear here after pest identification..."
                )
            
            # Right Column: Conversational Chat
            with gr.Column(scale=1):
                gr.Markdown("### 💬 Chat with Dr. GreenThumb", elem_classes="section-header")
                
                # Current Pest Context Display
                pest_context_display = gr.Markdown(
                    "**No pest identified yet.** Upload an image first, then ask questions!",
                    elem_classes="pest-context"
                )
                
                # Chat Interface
                chatbot = gr.Chatbot(
                    label="Conversation with Dr. GreenThumb",
                    height=350,
                    placeholder="👋 Hi! I'm Dr. GreenThumb, your organic farming consultant. Upload a pest image first, then ask me anything about organic pest management!"
                )
                
                # Chat Input
                chat_input = gr.Textbox(
                    label="Ask Dr. GreenThumb",
                    placeholder="Ask about treatments, timing, costs, alternatives...",
                    lines=2
                )
                
                with gr.Row():
                    send_btn = gr.Button("💬 Send", variant="primary")
                    clear_chat_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
                
                # Quick Question Buttons
                with gr.Column(elem_classes="quick-questions"):
                    gr.Markdown("**Quick Questions:**")
                    with gr.Row():
                        cost_btn = gr.Button("💰 Cost?", size="sm")
                        timing_btn = gr.Button("⏰ When to apply?", size="sm")
                        alternatives_btn = gr.Button("🔄 Alternatives?", size="sm")
                    with gr.Row():
                        prevention_btn = gr.Button("🛡️ Prevention?", size="sm")
                        safety_btn = gr.Button("⚠️ Safety tips?", size="sm")
                        followup_btn = gr.Button("📋 Follow-up?", size="sm")
        
        # Chat Export Section
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📄 Session Management")
                with gr.Row():
                    export_btn = gr.Button("📥 Export Chat History", variant="secondary")
                    new_session_btn = gr.Button("🆕 New Session", variant="secondary")
                
                chat_export = gr.Textbox(
                    label="Chat History Export",
                    lines=5,
                    visible=False,
                    interactive=False
                )
        
        # State variables
        current_pest = gr.State(None)
        current_confidence = gr.State(None)
        
        # Event Handlers
        def identify_pest(image):
            """Handle pest identification"""
            if image is None:
                return (
                    "❌ Please upload an image first.",
                    "No treatment recommendations available.",
                    None,
                    None,
                    "**No pest identified yet.** Upload an image first, then ask questions!"
                )
            
            try:
                # Get prediction
                identification_result, treatment = predictor.predict_pest(image)
                
                # Handle error cases first
                if any(error_key in identification_result for error_key in ['error', 'no_image', 'model_not_loaded']):
                    return (
                        identification_result,
                        treatment,
                        None,
                        None,
                        "**Unable to identify pest.** Please check the error message above."
                    )
                
                # Extract pest class and confidence from successful identification
                if 'species' in identification_result.lower():
                    # Parse the identification result text to extract pest class and confidence
                    lines = identification_result.split('\n')
                    pest_class = None
                    confidence = None
                    
                    for line in lines:
                        line_lower = line.lower()
                        if 'species:' in line_lower:
                            # Handle both **Species:** and Species: formats
                            pest_class = line.split(':')[1].strip().replace('*', '').lower()
                        elif 'confidence:' in line_lower:
                            # Handle both **Confidence:** and Confidence: formats
                            confidence_str = line.split(':')[1].strip().replace('*', '').replace('%', '')
                            confidence = float(confidence_str) / 100
                    
                    if pest_class and confidence:
                        # Update chat service with pest context
                        chat_service.add_pest_context(pest_class, confidence)
                        
                        # Format identification result
                        identification_text = identification_result
                        
                        # Update pest context display
                        context_display = f"""🎯 **Current Pest:** {pest_class.title()} ({confidence:.1%} confidence)
💬 **Ask me anything about treating this pest!**"""
                        
                        return (
                            identification_text,
                            treatment,
                            pest_class,
                            confidence,
                            context_display
                        )
                
                # Handle uncertain or unclear cases
                return (
                    identification_result,
                    treatment,
                    None,
                    None,
                    "**Uncertain identification.** Please try another image."
                )
                    
            except Exception as e:
                return (
                    f"❌ Error during identification: {str(e)}",
                    "Unable to provide treatment recommendations.",
                    None,
                    None,
                    "**Error occurred.** Please try again."
                )
        
        def send_chat_message(message, chat_history, pest_class, confidence):
            """Handle chat message sending"""
            if not message.strip():
                return chat_history, ""
            
            # Add user message to chat
            chat_history = chat_history or []
            chat_history.append([message, None])
            
            # Get AI response
            try:
                ai_response = chat_service.chat(message)
                chat_history[-1][1] = ai_response
            except Exception as e:
                chat_history[-1][1] = f"❌ Sorry, I encountered an error: {str(e)}"
            
            return chat_history, ""
        
        def send_quick_question(question, chat_history, pest_class, confidence):
            """Handle quick question buttons"""
            return send_chat_message(question, chat_history, pest_class, confidence)
        
        def clear_chat(pest_class, confidence):
            """Clear chat history but maintain pest context"""
            if pest_class:
                chat_service.start_new_session(pest_class, confidence)
                return [], f"🔄 Chat cleared. Current pest: {pest_class.title()}"
            else:
                chat_service.start_new_session()
                return [], "🔄 Chat cleared. Upload an image to identify a pest first!"
        
        def new_session():
            """Start completely new session"""
            chat_service.start_new_session()
            return (
                [],
                "🆕 New session started! Upload an image to begin.",
                None,
                None,
                "**No pest identified yet.** Upload an image first, then ask questions!",
                "❌ Please upload an image first.",
                "No treatment recommendations available."
            )
        
        def export_chat_history():
            """Export current chat history"""
            try:
                history = chat_service.export_chat_history()
                return gr.update(value=history, visible=True)
            except Exception as e:
                return gr.update(value=f"Error exporting chat: {str(e)}", visible=True)
        
        # Wire up events
        identify_btn.click(
            identify_pest,
            inputs=[image_input],
            outputs=[identification_output, treatment_output, current_pest, current_confidence, pest_context_display]
        )
        
        # Chat events
        send_btn.click(
            send_chat_message,
            inputs=[chat_input, chatbot, current_pest, current_confidence],
            outputs=[chatbot, chat_input]
        )
        
        chat_input.submit(
            send_chat_message,
            inputs=[chat_input, chatbot, current_pest, current_confidence],
            outputs=[chatbot, chat_input]
        )
        
        # Quick question buttons
        cost_btn.click(
            lambda ch, p, c: send_quick_question("What are the costs for treating this pest?", ch, p, c),
            inputs=[chatbot, current_pest, current_confidence],
            outputs=[chatbot, chat_input]
        )
        
        timing_btn.click(
            lambda ch, p, c: send_quick_question("When is the best time to apply these treatments?", ch, p, c),
            inputs=[chatbot, current_pest, current_confidence],
            outputs=[chatbot, chat_input]
        )
        
        alternatives_btn.click(
            lambda ch, p, c: send_quick_question("What are some alternative organic treatments?", ch, p, c),
            inputs=[chatbot, current_pest, current_confidence],
            outputs=[chatbot, chat_input]
        )
        
        prevention_btn.click(
            lambda ch, p, c: send_quick_question("How can I prevent this pest in the future?", ch, p, c),
            inputs=[chatbot, current_pest, current_confidence],
            outputs=[chatbot, chat_input]
        )
        
        safety_btn.click(
            lambda ch, p, c: send_quick_question("What safety precautions should I take?", ch, p, c),
            inputs=[chatbot, current_pest, current_confidence],
            outputs=[chatbot, chat_input]
        )
        
        followup_btn.click(
            lambda ch, p, c: send_quick_question("What should I do for follow-up monitoring?", ch, p, c),
            inputs=[chatbot, current_pest, current_confidence],
            outputs=[chatbot, chat_input]
        )
        
        # Session management
        clear_chat_btn.click(
            clear_chat,
            inputs=[current_pest, current_confidence],
            outputs=[chatbot, pest_context_display]
        )
        
        new_session_btn.click(
            new_session,
            outputs=[chatbot, pest_context_display, current_pest, current_confidence, pest_context_display, identification_output, treatment_output]
        )
        
        export_btn.click(
            export_chat_history,
            outputs=[chat_export]
        )
    
    return interface

if __name__ == "__main__":
    # Create and launch the conversational interface
    interface = create_conversational_pest_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )