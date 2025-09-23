from utils import TASKS
from feedback.qwen_utils import inference_from_frames, _extract_json_and_sentence,_load_recent_history, _append_memory, _summarize_history_for_prompt  
import torch
import numpy as np
import json, re, textwrap

def prompt_model(config, success, historical_context=None):
    """
    Generates a "Chain-of-Thought" prompt to encourage deeper analysis.
    """
    outcome = "succeeded" if success else "failed"
    
    history_prompt_section = ""
    if historical_context:
        history_prompt_section += "\n=== Historical Context ===\n"
        history_prompt_section += "Here is the feedback from similar past attempts:\n"
        for item in historical_context:
            history_prompt_section += f"- Past Outcome: {item['outcome']}\n  Past Feedback: \"{item['feedback']}\"\n"

    prompt = f"""
    You are an expert robot task evaluator providing reflective, chain-of-thought feedback.

    ### Task Name: {config.env_name}
    ### Task Instruction: {TASKS[config.env_name]}
    {history_prompt_section}
    ### Current Episode Analysis
    The agent {outcome} in the most recent attempt. First, think step-by-step about the key phases of the trajectory you are observing. Then, based on your analysis, provide a final, single-sentence summary of the primary reason for the outcome.

    **Your thought process should be:**
    1.  **Initial Approach:** Describe the agent's initial movement. Did it move towards the correct object?
    2.  **Key Interaction:** Analyze the moment of interaction. Did the gripper align correctly? What was the result of the grasp/push/etc.?
    3.  **Causal Reason:** Based on the above, conclude the single most important reason for the success or failure.

    **Final Answer Format:**

    Final Answer: [Your single-sentence summary here, starting with "Yes,..." or "No,..."]
    """
    return prompt

FEW_SHOTS = [
# Failure: button press from top, approached from side
{
  "instruction": "press a button from the top",
  "outcome":"failure",
  "primary_error":{"code":"bad_approach_direction","explanation":"The gripper came from the side, sliding off the button instead of a vertical press."},
  "secondary_factors":["insufficient_push_pull"],
  "key_frame_indices":[18, 22],
  "suggested_fix":"Approach from directly above the button; align gripper normal to the button surface, then press straight down.",
  "confidence":0.85,
  "summary":"No, the agent failed because it approached from the side and slid off instead of pressing straight down."
},
# Success: drawer-open with clear handle grasp and pull
{
  "instruction":"open a drawer",
  "outcome":"success",
  "primary_error":{"code":"wrong_object","explanation":"(n/a for success)"},
  "secondary_factors":[],
  "key_frame_indices":[9, 27, 41],
  "suggested_fix":"(n/a)",
  "confidence":0.9,
  "summary":"Yes, the agent succeeded because it grasped the handle securely and pulled along the drawer's opening direction."
}
]

ERROR_TAXONOMY = [
    {"code": "wrong_object", "desc": "Interacted with the wrong object."},
    {"code": "bad_approach_direction", "desc": "Approached object from a wrong angle/direction."},
    {"code": "failed_grasp", "desc": "Contact without a stable grasp; slipped or never closed gripper appropriately."},
    {"code": "insufficient_force", "desc": "Touched correct object but did not exert proper motion/force."},
    {"code": "drift_from_goal", "desc": "Trajectories drifted away from the goal, no course correction."}
]

JSON_SPEC = textwrap.dedent("""
Return ONLY:
1) A single JSON object matching this JSON Schema:
{
  "type":"object",
  "properties":{
    "outcome":{"type":"string","enum":["success","failure"]},
    "primary_error":{"type":"object","properties":{
        "code":{"type":"string"},
        "explanation":{"type":"string"}
    }, "required":["code","explanation"]},
    "secondary_factors":{"type":"array","items":{"type":"string"}},
    "key_frame_indices":{"type":"array","items":{"type":"integer"}},
    "suggested_fix":{"type":"string"},
    "confidence":{"type":"number","minimum":0,"maximum":1}
  },
  "required":["outcome","primary_error","key_frame_indices","suggested_fix","confidence"]
}
2) Then on the next line ONLY - short, direct, to the point - single human sentence summary.
""").strip()

def build_feedback_prompt(config, success: bool, history_text: str) -> str:
    """
    Goal: Force a precise error class, ask for evidence, and ground it in specific frames.
    """
    taxo_lines = "\n".join([f"- {t['code']}: {t['desc']}" for t in ERROR_TAXONOMY])
    outcome = "success" if success else "failure"
    return textwrap.dedent(f"""
    You are a robot task evaluator analyzing a single MetaWorld episode (video frames are attached).

    Task name: {config.env_name}
    Instruction: {TASKS[config.env_name]}
    Episode outcome observed in env: {outcome}

    Recent history: {history_text}

    Use the error taxonomy below to identify the *primary* cause (choose ONE code), and optionally list secondary factors:
    {taxo_lines}

    Think carefully step by step about what happened in the episode. Identify the *critical moment* and cite 1-3 key frame indices from the provided sequence that best support your diagnosis. 
    Then produce:
    {JSON_SPEC}
    """).strip()

def _few_shots_text():
    lines = []
    for ex in FEW_SHOTS:
        js = {k:v for k,v in ex.items() if k != "summary"}
        lines.append("Example JSON:\n" + json.dumps(js, ensure_ascii=False))
        lines.append("Example one-liner:\n" + ex["summary"])
    return "\n\n".join(lines)

def qwen_feedback(model, processor, frames, config, success, model_path="Qwen/Qwen2.5-VL-3B-Instruct",
                  memory_path: str = None, exp_name: str = None):
    """
    Returns a dict with parsed JSON + 'summary' field for humans.
    Also appends to episodic memory if memory_path is provided.
    """
    # history
    hist = _load_recent_history(memory_path, config.env_name, k=10) if memory_path else []
    hist_text = _summarize_history_for_prompt(hist)
    # build prompt
    prompt = build_feedback_prompt(config, success, hist_text)
    # prepend few-shots as textual guidance
    fewshots = _few_shots_text()
    full_prompt = (
        "You will see a short video of the episode.\n"
        "First, think step by step (do NOT output your reasoning).\n"
        "Then output ONLY the requested JSON, and on the next line a single human sentence.\n\n"
        + fewshots + "\n\n=== Begin Episode Analysis ===\n" + prompt
    )

    # Retry until parsed structure is valid or max attempts hit
    for attempt in range(10):
        raw = inference_from_frames(frames, full_prompt, processor, model, max_new_tokens=2048, temperature=0.5)
        parsed = _extract_json_and_sentence(raw, config=config)

        # Validate structure (strict type check, not just key presence)
        if isinstance(parsed.get("outcome"), str) and \
        isinstance(parsed.get("primary_error"), dict) and \
        isinstance(parsed["primary_error"].get("code"), str) and \
        isinstance(parsed.get("secondary_factors"), list) and \
        isinstance(parsed.get("key_frame_indices"), list) and \
        isinstance(parsed.get("suggested_fix"), str) and \
        isinstance(parsed.get("confidence"), (float, int)) and \
        isinstance(parsed.get("summary"), str):
            break  # success
        else:
            pass
    # append memory
    if memory_path:
        rec = {
            "task": config.env_name,
            "outcome": parsed.get("outcome"),
            "primary_error": parsed.get("primary_error"),
            "secondary_factors": parsed.get("secondary_factors", []),
            "key_frame_indices": parsed.get("key_frame_indices", []),
            "suggested_fix": parsed.get("suggested_fix", ""),
            "confidence": parsed.get("confidence", 0.0),
            "summary": parsed.get("summary", ""),
        }
        _append_memory(memory_path, rec)
    return parsed
