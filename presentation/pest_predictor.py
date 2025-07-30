"""
Pest Prediction Logic
Handles AI model loading and prediction processing
"""

import numpy as np
import json
import sys
from pathlib import Path
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.config import *

class PestPredictor:
    """
    Handles pest identification and treatment recommendations
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
            # Try to load TensorFlow and model
            import tensorflow as tf
            
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
                
        except ImportError:
            print("❌ TensorFlow not available - running in demo mode")
            self.model = None
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
            return "✅ **No treatment needed** - This appears to be a beneficial insect!"
        
        # Get pest characteristics
        economic_impact = ECONOMIC_IMPACT.get(pest_class, 3)
        urgency = TREATMENT_URGENCY.get(pest_class, 'Medium')
        
        # Base organic treatments by pest type
        treatments = {
            'beetle': "🌿 **Organic Treatments:**\\n• Neem oil spray\\n• Beneficial nematodes\\n• Row covers\\n• Hand picking\\n• Diatomaceous earth",
            'catterpillar': "🌿 **Organic Treatments:**\\n• Bacillus thuringiensis (Bt)\\n• Row covers\\n• Hand picking\\n• Beneficial wasps\\n• Companion planting with herbs",
            'earwig': "🌿 **Organic Treatments:**\\n• Beer traps\\n• Diatomaceous earth\\n• Remove garden debris\\n• Beneficial predators\\n• Copper strips",
            'grasshopper': "🌿 **Organic Treatments:**\\n• Row covers\\n• Beneficial birds habitat\\n• Neem oil\\n• Kaolin clay spray\\n• Timing of plantings",
            'moth': "🌿 **Organic Treatments:**\\n• Pheromone traps\\n• Bacillus thuringiensis (Bt)\\n• Row covers during flight season\\n• Beneficial parasitic wasps\\n• Light traps",
            'slug': "🌿 **Organic Treatments:**\\n• Beer traps\\n• Copper barriers\\n• Diatomaceous earth\\n• Iron phosphate baits\\n• Remove hiding places",
            'snail': "🌿 **Organic Treatments:**\\n• Beer traps\\n• Copper barriers\\n• Diatomaceous earth\\n• Hand picking\\n• Crushed eggshells",
            'wasp': "🌿 **Organic Treatments:**\\n• Usually beneficial! Only treat if problematic\\n• Remove food sources\\n• Seal nest entrances\\n• Professional removal if needed",
            'weevil': "🌿 **Organic Treatments:**\\n• Beneficial nematodes\\n• Diatomaceous earth\\n• Remove infected plants\\n• Crop rotation\\n• Sticky traps"
        }
        
        treatment = treatments.get(pest_class, "Consult local agricultural extension for specific recommendations.")
        
        # Add urgency and impact information
        impact_emojis = {1: "💚", 2: "💛", 3: "🧡", 4: "❤️", 5: "💔"}
        urgency_emojis = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
        
        recommendation = f"""**{urgency_emojis[urgency]} Urgency Level: {urgency}**
**{impact_emojis[economic_impact]} Economic Impact: {economic_impact}/5**

{treatment}

**🚨 Action Priority:**
{'🔥 **IMMEDIATE ACTION REQUIRED**' if urgency == 'High' else 
 '⚡ **ACTION RECOMMENDED**' if urgency == 'Medium' else 
 '📝 **MONITOR SITUATION**'}"""
        
        return recommendation
    
    def predict_pest(self, image):
        """
        Main prediction function called by the interface
        """
        if image is None:
            return "📸 **Please upload an image**", "Upload a clear photo of the pest for identification."
        
        if self.model is None:
            return ("❌ **Model not loaded**\\n\\nPlease train the model first using:\\n"
                   "1. `python src/data_preprocessing.py`\\n"
                   "2. `python src/model_training.py`"), ("**Demo Mode**\\n\\n"
                   "The AI model needs to be trained before making predictions. "
                   "Once trained, you'll get real pest identification and treatment recommendations.")
        
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
                result = f"🤔 **Uncertain Identification**\\n\\nMost likely: {predicted_class.title()}\\nConfidence: {confidence_percent:.1f}%\\n\\n⚠️ Low confidence - consider taking a clearer photo"
                recommendation = "**Recommendation:** Take a clearer, closer photo in good lighting for better identification."
            else:
                result = f"🔍 **Pest Identified**\\n\\n**Species:** {predicted_class.title()}\\n**Confidence:** {confidence_percent:.1f}%"
                recommendation = self.get_treatment_recommendation(predicted_class)
            
            return result, recommendation
            
        except Exception as e:
            return f"❌ **Error during prediction:** {str(e)}", "Please try again with a different image."
