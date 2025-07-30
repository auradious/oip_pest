"""
Gradio Frontend for Organic Farm Pest Management AI System
Web interface for farmers to upload pest images and get treatment recommendations
"""

import gradio as gr
import numpy as np
import json
import tensorflow as tf
from PIL import Image
from pathlib import Path
import sys

# Import configuration
sys.path.append(str(Path(__file__).parent.parent))
from config.config import *

class PestIdentificationApp:
    """
    Gradio web application for pest identification and treatment recommendations
    """
    
    def __init__(self):
        self.model = None
        self.class_names = {}
        self.load_model_and_classes()
        
    def load_model_and_classes(self):
        """
        Load the trained model and class mappings
        """
        try:
            # Load trained model
            model_path = MODEL_PATHS['best_model']
            if model_path.exists():
                self.model = tf.keras.models.load_model(model_path)
                print(f"✅ Model loaded from {model_path}")
            else:
                print(f"❌ Model not found at {model_path}")
                self.model = None
                
            # Load class mappings
            class_names_path = MODEL_PATHS['class_names']
            if class_names_path.exists():
                with open(class_names_path, 'r') as f:
                    self.class_names = json.load(f)
                print(f"✅ Class names loaded from {class_names_path}")
            else:
                print(f"❌ Class names not found at {class_names_path}")
                # Fallback to config classes
                self.class_names = {
                    'idx_to_class': {str(i): cls for i, cls in enumerate(HARMFUL_PEST_CLASSES)}
                }
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
    
    def preprocess_image(self, image):
        """
        Preprocess uploaded image for model prediction
        """
        if image is None:
            return None
            
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Resize to model input size
        target_size = IMAGE_CONFIG['target_size']
        image = image.resize(target_size)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array and normalize
        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array.astype('float32') / 255.0  # Normalize
        
        return img_array
    
    def get_treatment_recommendation(self, pest_class):
        """
        Get organic treatment recommendations for identified pest
        """
        if pest_class not in HARMFUL_PEST_CLASSES:
            return "No treatment needed - This appears to be a beneficial insect!"
        
        # Get pest characteristics
        economic_impact = ECONOMIC_IMPACT.get(pest_class, 3)
        urgency = TREATMENT_URGENCY.get(pest_class, 'Medium')
        
        # Base organic treatments by pest type
        treatments = {
            'beetle': "🌿 **Organic Treatments:**\n• Neem oil spray\n• Beneficial nematodes\n• Row covers\n• Hand picking\n• Diatomaceous earth",
            'catterpillar': "🌿 **Organic Treatments:**\n• Bacillus thuringiensis (Bt)\n• Row covers\n• Hand picking\n• Beneficial wasps\n• Companion planting with herbs",
            'earwig': "🌿 **Organic Treatments:**\n• Beer traps\n• Diatomaceous earth\n• Remove garden debris\n• Beneficial predators\n• Copper strips",
            'grasshopper': "🌿 **Organic Treatments:**\n• Row covers\n• Beneficial birds habitat\n• Neem oil\n• Kaolin clay spray\n• Timing of plantings",
            'moth': "🌿 **Organic Treatments:**\n• Pheromone traps\n• Bacillus thuringiensis (Bt)\n• Row covers during flight season\n• Beneficial parasitic wasps\n• Light traps",
            'slug': "🌿 **Organic Treatments:**\n• Beer traps\n• Copper barriers\n• Diatomaceous earth\n• Iron phosphate baits\n• Remove hiding places",
            'snail': "🌿 **Organic Treatments:**\n• Beer traps\n• Copper barriers\n• Diatomaceous earth\n• Hand picking\n• Crushed eggshells",
            'wasp': "🌿 **Organic Treatments:**\n• Usually beneficial! Only treat if problematic\n• Remove food sources\n• Seal nest entrances\n• Professional removal if needed",
            'weevil': "🌿 **Organic Treatments:**\n• Beneficial nematodes\n• Diatomaceous earth\n• Remove infected plants\n• Crop rotation\n• Sticky traps"
        }
        
        treatment = treatments.get(pest_class, "Consult local agricultural extension for specific recommendations.")
        
        # Add urgency and impact information
        impact_emojis = {1: "💚", 2: "💛", 3: "🧡", 4: "❤️", 5: "💔"}
        urgency_emojis = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
        
        recommendation = f"""
**{urgency_emojis[urgency]} Urgency Level: {urgency}**
**{impact_emojis[economic_impact]} Economic Impact: {economic_impact}/5**

{treatment}

**🚨 Action Priority:**
{'🔥 **IMMEDIATE ACTION REQUIRED**' if urgency == 'High' else 
 '⚡ **ACTION RECOMMENDED**' if urgency == 'Medium' else 
 '📝 **MONITOR SITUATION**'}
        """
        
        return recommendation
    
    def predict_pest(self, image):
        """
        Main prediction function called by Gradio interface
        """
        if self.model is None:
            return "❌ **Model not loaded!** Please train the model first.", "No recommendations available."
        
        if image is None:
            return "📸 **Please upload an image**", "Upload a clear photo of the pest for identification."
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            
            # Get class name
            predicted_class = self.class_names['idx_to_class'].get(str(predicted_class_idx), 'Unknown')
            
            # Format prediction result
            confidence_percent = confidence * 100
            
            if confidence < 0.3:
                result = f"🤔 **Uncertain Identification**\n\nMost likely: {predicted_class.title()}\nConfidence: {confidence_percent:.1f}%\n\n⚠️ Low confidence - consider taking a clearer photo"
                recommendation = "**Recommendation:** Take a clearer, closer photo in good lighting for better identification."
            else:
                result = f"🔍 **Pest Identified**\n\n**Species:** {predicted_class.title()}\n**Confidence:** {confidence_percent:.1f}%"
                recommendation = self.get_treatment_recommendation(predicted_class)
            
            return result, recommendation
            
        except Exception as e:
            return f"❌ **Error during prediction:** {str(e)}", "Please try again with a different image."

# Initialize the app
app = PestIdentificationApp()

def create_interface():
    """
    Create and configure the Gradio interface
    """
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title=GRADIO_CONFIG['title'],
        css="""
        .gradio-container {
            max-width: 800px !important;
            margin: auto !important;
        }
        """
    ) as interface:
        
        gr.Markdown(f"# {GRADIO_CONFIG['title']}")
        gr.Markdown(GRADIO_CONFIG['description'])
        
        with gr.Row():
            with gr.Column(scale=1):
                # Image input
                image_input = gr.Image(
                    label="📸 Upload Pest Image",
                    type="pil",
                    height=400
                )
                
                # Predict button
                predict_btn = gr.Button(
                    "🔍 Identify Pest", 
                    variant="primary",
                    size="lg"
                )
                
                # Example images (if available)
                gr.Markdown("### 📝 Tips for best results:")
                gr.Markdown("""
                • Take clear, close-up photos
                • Ensure good lighting
                • Focus on the pest, not the background
                • Multiple angles can help accuracy
                """)
            
            with gr.Column(scale=1):
                # Results
                prediction_output = gr.Textbox(
                    label="🎯 Identification Result",
                    lines=6,
                    interactive=False
                )
                
                treatment_output = gr.Textbox(
                    label="🌿 Treatment Recommendations",
                    lines=12,
                    interactive=False
                )
        
        # Connect the prediction function
        predict_btn.click(
            fn=app.predict_pest,
            inputs=[image_input],
            outputs=[prediction_output, treatment_output]
        )
        
        # Footer
        gr.Markdown("""
        ---
        **🌱 Organic Farm Pest Management AI** | 
        Helping farmers protect crops while preserving beneficial insects | 
        *Always consult local agricultural experts for comprehensive pest management*
        """)
    
    return interface

if __name__ == "__main__":
    # Create and launch the interface
    interface = create_interface()
    
    print(f"🚀 Starting {GRADIO_CONFIG['title']}...")
    print(f"🌐 Server will run on port {GRADIO_CONFIG['server_port']}")
    
    interface.launch(
        server_port=GRADIO_CONFIG['server_port'],
        server_name=GRADIO_CONFIG['server_name'],
        share=GRADIO_CONFIG['share'],
        debug=GRADIO_CONFIG['debug']
    )
