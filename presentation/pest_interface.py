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
from config.languages import LANGUAGES, DEFAULT_LANGUAGE, AVAILABLE_LANGUAGES
from presentation.pest_predictor import PestPredictor

def create_pest_management_interface():
    """
    Create and configure the main Gradio interface with multilingual support
    """
    # Initialize the predictor
    predictor = PestPredictor()
    
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title=GRADIO_CONFIG['title'],
        css="""
        .gradio-container {
            max-width: 1000px !important;
            margin: auto !important;
            padding: 20px;
        }
        .upload-container {
            border: 2px dashed #ccc;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            min-height: 400px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .result-container {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            min-height: 120px;
        }
        .language-selector {
            position: absolute;
            top: 20px;
            right: 20px;
            z-index: 1000;
            min-width: 150px;
        }
        .main-content {
            margin-top: 60px;
        }
        .section-header {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 15px;
            min-height: 30px;
            display: flex;
            align-items: center;
        }
        .tips-section {
            margin-top: 20px;
        }
        .footer-section {
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #eee;
        }
        .column-left, .column-right {
            padding: 0 15px;
        }
        .button-container {
            margin: 20px 0;
        }
        """
    ) as interface:
        
        # Language selector in top right
        with gr.Row():
            with gr.Column(scale=4):
                pass  # Empty space
            with gr.Column(scale=1, elem_classes="language-selector"):
                language_selector = gr.Dropdown(
                    choices=AVAILABLE_LANGUAGES,
                    value='en',
                    label="🌐",
                    show_label=False,
                    container=False,
                    scale=1
                )
        
        # Main content with proper spacing
        with gr.Column(elem_classes="main-content"):
            # Dynamic content that changes with language
            title_md = gr.Markdown(f"# {LANGUAGES[DEFAULT_LANGUAGE]['ui']['title']}")
            description_md = gr.Markdown(LANGUAGES[DEFAULT_LANGUAGE]['ui']['description'])
            gr.Markdown("---")
            
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, elem_classes="column-left"):
                    # Image Upload Section
                    upload_section_md = gr.Markdown(
                        f"### {LANGUAGES[DEFAULT_LANGUAGE]['ui']['upload_section']}",
                        elem_classes="section-header"
                    )
                    image_input = gr.Image(
                        label=LANGUAGES[DEFAULT_LANGUAGE]['ui']['upload_label'],
                        type="pil",
                        height=350,
                        elem_classes="upload-container"
                    )
                    
                    # Action Button
                    with gr.Row(elem_classes="button-container"):
                        predict_btn = gr.Button(
                            LANGUAGES[DEFAULT_LANGUAGE]['ui']['identify_button'], 
                            variant="primary",
                            size="lg",
                            scale=1
                        )
                    
                    # Tips Section
                    with gr.Column(elem_classes="tips-section"):
                        with gr.Accordion(LANGUAGES[DEFAULT_LANGUAGE]['ui']['tips_title'], open=False) as tips_accordion:
                            tips_content = gr.Markdown(LANGUAGES[DEFAULT_LANGUAGE]['ui']['tips_content'])
                
                with gr.Column(scale=1, elem_classes="column-right"):
                    # Results Section
                    results_section_md = gr.Markdown(
                        f"### {LANGUAGES[DEFAULT_LANGUAGE]['ui']['results_section']}",
                        elem_classes="section-header"
                    )
                    
                    identification_output = gr.Textbox(
                        label=LANGUAGES[DEFAULT_LANGUAGE]['ui']['identification_label'],
                        lines=4,
                        interactive=False,
                        elem_classes="result-container"
                    )
                    
                    treatment_output = gr.Textbox(
                        label=LANGUAGES[DEFAULT_LANGUAGE]['ui']['treatment_label'],
                        lines=10,
                        interactive=False,
                        elem_classes="result-container"
                    )
        
            # Footer
            with gr.Column(elem_classes="footer-section"):
                footer_md = gr.Markdown(f"""
                <div style='text-align: center; color: #666; font-size: 14px; padding: 20px;'>
                <p><strong>{LANGUAGES[DEFAULT_LANGUAGE]['ui']['footer_title']}</strong></p>
                <p>{LANGUAGES[DEFAULT_LANGUAGE]['ui']['footer_subtitle']}</p>
                <p><em>{LANGUAGES[DEFAULT_LANGUAGE]['ui']['footer_disclaimer']}</em></p>
                </div>
                """)
        
        # Function to update UI language
        def update_language(selected_language):
            lang_data = LANGUAGES.get(selected_language, LANGUAGES[DEFAULT_LANGUAGE])
            
            return [
                f"# {lang_data['ui']['title']}",  # title_md
                lang_data['ui']['description'],  # description_md
                f"### {lang_data['ui']['upload_section']}",  # upload_section_md
                gr.update(label=lang_data['ui']['upload_label']),  # image_input
                lang_data['ui']['identify_button'],  # predict_btn
                gr.update(label=lang_data['ui']['tips_title']),  # tips_accordion
                lang_data['ui']['tips_content'],  # tips_content
                f"### {lang_data['ui']['results_section']}",  # results_section_md
                gr.update(label=lang_data['ui']['identification_label']),  # identification_output
                gr.update(label=lang_data['ui']['treatment_label']),  # treatment_output
                f"""
                <div style='text-align: center; color: #666; font-size: 14px; padding: 20px;'>
                <p><strong>{lang_data['ui']['footer_title']}</strong></p>
                <p>{lang_data['ui']['footer_subtitle']}</p>
                <p><em>{lang_data['ui']['footer_disclaimer']}</em></p>
                </div>
                """  # footer_md
            ]
        
        # Function to predict with language support
        def predict_with_language(image, language):
            return predictor.predict_pest(image, language)
        
        # Connect language selector to update UI
        language_selector.change(
            fn=update_language,
            inputs=[language_selector],
            outputs=[
                title_md, description_md, upload_section_md, image_input, 
                predict_btn, tips_accordion, tips_content, results_section_md,
                identification_output, treatment_output, footer_md
            ]
        )
        
        # Connect the prediction function with language support
        predict_btn.click(
            fn=predict_with_language,
            inputs=[image_input, language_selector],
            outputs=[identification_output, treatment_output]
        )
    
    return interface
