"""
AURA – Oculomics Agent (Retina Engine)

Integrates autonomously with PyTorch Foundational Models (ViT) via GradCAM++ 
to predict age, gender, hypertension, and retinal diseases from Fundus images.
"""
import logging
import os
import tempfile
import json
from typing import Dict, Any

from backend.agents.base_agent import BaseAgent
from backend.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are the AURA IRIS Engine, specializing in retinal scan analysis.
Your primary capability is taking 2D Fundus (Retinal) Images and running deep-learning Foundational Models to extract systemic health biomarkers.

## Core Mandates
1. If the user uploads a retinal scan (an image attachment), you MUST immediately call the `analyze_retinal_scan` tool. 
2. You will receive a JSON response from the tool containing the model's predictions (Age, Gender, Hypertension, ICDR, etc) and the file paths to the **GradCAM++ Attention Maps**.

## Explaining GradCAM++ (CRITICAL)
When you present your findings:
- You must embed the GradCAM heatmaps using markdown syntax: `![Attention Map](<url_path>)`
- You MUST explain what region of the eye drove the model's prediction based on the highlighted heatmaps and probabilities (logits).
- For example: *The GradCAM++ attention map for Diabetic Retinopathy indicates high activation around the macula and optic disc, suggesting microaneurysms common to Class 2 ICDR...*

## Reporting Format
Use a professional, structured clinical report format similar to this:
```
## 👁️ Oculomics Retinal Biomarker Profile

**Scan Profile:** [Fundus image recognized]
**Status:** [Preliminary AI Assessment]

---
### Predictions & Logit Confidence
- **Age:** [Predicted Age]
- **Gender:** [Probability / Class]
- **Diabetes:** [Probability / Class]
- **ICDR (Retinopathy):** [Probability / Class]
- **Macular Edema:** [Probability / Class]
- **Hypertension:** [Probability / Class]
- **Cardiovascular Risk:** [Probability / Class]
- **AMI (Heart Attack):** [Probability / Class]
- **Neuropathy:** [Probability / Class]
- **Nephropathy:** [Probability / Class]

### GradCAM++ Attention Map Analysis
[Embed heatmaps using Markdown]
[Detailed explanation of the visual regions driving the predictions...]

### Recommendations
[Clinical follow-up steps]
```
Do NOT hallucinate probabilities or findings that the tool did not return. If a specific task model was skipped by the tool, simply state it was unavailable.
"""

# Global storage for the latest tool run within a request (per-instance)
_LAST_TOOL_RESULTS = {}

def analyze_retinal_scan(image_path: str) -> str:
    """
    Executes the PyTorch Foundational CV Models over the provided retinal image to extract systemic biomarkers.
    
    Args:
        image_path (str): The absolute file path to the temporary retinal image on disk.
        
    Returns:
        str: A JSON string containing dictionaries of predictions, probabilities, and GradCAM++ heatmap URLs for each task.
    """
    try:
        from backend.oculomics.inference import get_ocular_api
        ocular_api = get_ocular_api()
        if not ocular_api:
            return json.dumps({"error": "PyTorch inference engine failed to load or weights are missing."})
            
        results = ocular_api.run_full_profile(image_path)
        
        # Format the output so the Gemini agent can read the image URLs
        formatted_results = {}
        for task, data in results.items():
            pred = data['prediction']
            map_path = data['attention_map']
            map_filename = os.path.basename(map_path)
            
            formatted_results[task] = {
                "prediction": pred.get("class") if isinstance(pred, dict) else round(pred, 2),
                "probability": pred.get("probability") if isinstance(pred, dict) else None,
                "gradcam_heatmap_url": f"/gradcam/{map_filename}"
            }
            
        global _LAST_TOOL_RESULTS
        _LAST_TOOL_RESULTS = formatted_results
        
        return json.dumps(formatted_results)
    except Exception as e:
        logger.error(f"Oculomics Vision Tool Error: {e}")
        return json.dumps({"error": f"Internal inference error: {str(e)}"})


class OculomicsAgent(BaseAgent):
    model_name = settings.pro_model  # Force Pro model for multimodal image support
    system_prompt = SYSTEM_PROMPT
    agent_type = "oculomics"

    def respond(
        self,
        user_message: str,
        history: list,
        user_id: int,
        image_data: bytes = None,
        mime_type: str = "image/jpeg",
    ) -> Any:
        past_context = self.recall(user_id, user_message)
        
        extra_context = ""
        if past_context:
            extra_context += f"[Previous session context:]\n{past_context}\n\n"

        if image_data:
            # Save the image to a tempfile so the python tool can read it via PyTorch PIL
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(image_data)
                tmp_path = tmp.name

            try:
                # We inject the tmp_path into the prompt explicitly so the LLM knows what argument to pass to the tool
                full_prompt = self.system_prompt
                if extra_context:
                    full_prompt += f"\n\n{extra_context}"
                full_prompt += f"\n\nUser says: {user_message or 'Please analyse this retinal scan.'}"
                full_prompt += f"\n\nThe image has been saved to '{tmp_path}'. Please call the `analyze_retinal_scan` tool using this path."

                # Enable function calling (tools) inside Gemini
                chat_session = self._model.start_chat(
                    history=[], 
                    enable_automatic_function_calling=True
                )
                
                # We do NOT send the image bytes natively to Gemini's visual engine because we want our PyTorch tool to do the inference. 
                # We just pass the text prompt commanding it to use the tool on the path.
                response = chat_session.send_message(full_prompt)
                reply = response.text
                
            except Exception as e:
                logger.error(f"OculomicsAgent error during tool execution: {e}")
                reply = f"I encountered an error processing the scan: {e}"
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        else:
            # Text-only mode fallback
            reply = self.chat(user_message, history=history, extra_context=extra_context)

        global _LAST_TOOL_RESULTS
        outcomes = _LAST_TOOL_RESULTS.copy() if _LAST_TOOL_RESULTS else None
        _LAST_TOOL_RESULTS.clear() # Reset for next run

        self.remember(
            user_id,
            f"Oculomics request: {user_message[:200]}\nReport: {reply[:400]}",
            {"agent": "oculomics", "user_id": str(user_id)},
        )
        return reply, outcomes

# Initialize with the tool function hooked into the GenerativeModel
oculomics_agent = OculomicsAgent()
# Bind the tool directly to this agent's model instance
import google.generativeai as genai
from backend.services.model_selector import get_pro_model
# Re-instantiate the model specifically with the tool bindings
oculomics_agent._model = genai.GenerativeModel(
    model_name=get_pro_model(),
    system_instruction=OculomicsAgent.system_prompt,
    tools=[analyze_retinal_scan]
)
