"""
Gradio Interface for Organic Farm Pest Management AI System
Clean, modular frontend interface
"""

import gradio as gr
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.config import GRADIO_CONFIG
from presentation.pest_predictor import PestPredictor

def create_pest_management_interface():
    """
    Create and configure the main Gradio interface
    """
    # Initialize the predictor
    predictor = PestPredictor()
    
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title=GRADIO_CONFIG['title'],
        css="""
        .gradio-container {
            max-width: 900px !important;
            margin: auto !important;
        }
        .upload-container {
            border: 2px dashed #ccc;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }
        .result-container {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
        }
        """
    ) as interface:
        
        # Header
        gr.Markdown(f"# {GRADIO_CONFIG['title']}")
        gr.Markdown(GRADIO_CONFIG['description'])
        gr.Markdown("---")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Image Upload Section
                gr.Markdown("### 📸 Upload Pest Image")
                image_input = gr.Image(
                    label="Drag and drop or click to upload",
                    type="pil",
                    height=350,
                    elem_classes="upload-container"
                )
                
                # Action Button
                predict_btn = gr.Button(
                    "🔍 Identify Pest", 
                    variant="primary",
                    size="lg",
                    scale=1
                )
                
                # Tips Section
                with gr.Accordion("📝 Photography Tips", open=False):
                    gr.Markdown("""
                    **For best identification results:**
                    - Take clear, close-up photos
                    - Ensure good lighting
                    - Focus on the pest, minimize background
                    - Multiple angles can help accuracy
                    - Avoid blurry or dark images
                    """)
            
            with gr.Column(scale=1):
                # Results Section
                gr.Markdown("### 🎯 Identification Results")
                
                identification_output = gr.Textbox(
                    label="Pest Identification",
                    lines=4,
                    interactive=False,
                    elem_classes="result-container"
                )
                
                treatment_output = gr.Textbox(
                    label="🌿 Treatment Recommendations",
                    lines=10,
                    interactive=False,
                    elem_classes="result-container"
                )
        
        # Connect the prediction function
        predict_btn.click(
            fn=predictor.predict_pest,
            inputs=[image_input],
            outputs=[identification_output, treatment_output]
        )
        
        # Footer
        gr.Markdown("---")
        gr.Markdown("""
        <div style='text-align: center; color: #666; font-size: 14px;'>
        <p><strong>🌱 Organic Farm Pest Management AI</strong></p>
        <p>Helping farmers protect crops while preserving beneficial insects</p>
        <p><em>Always consult local agricultural experts for comprehensive pest management</em></p>
        </div>
        """)
    
    return interface
