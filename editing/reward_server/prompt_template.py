SCORE_LOGIT = """Here are two images: the original and the edited version. Please evaluate the edited image based on the following editing instruction and requirement. 
Instruction: {prompt}
Requirements: {requirement}
You need to rate the editing result from 0 to 5 based on the accuracy and quality of the edit. 
0: The wrong object was edited, or the edit completely fails to meet the requirements. 
5: The correct object was edited, the requirements were met, and the visual result is high quality.
Response Format (Directly response the score number): 
0-5"""


SCORE_CONSISTENCY = """
You are a highly skilled image evaluator. You will receive two images (an original image and a modified image) and a specific edit instruction. The second image is known to have been altered based on this instruction, starting from the first image. Your task is to evaluate how well the second image is consistent with the original image.

Instruction: {prompt}

## Definitions

**Significant Change**: A noticeable alteration that substantially affects the visual perception or semantic content of the image. 

**Minor Change**: A subtle alteration that has limited impact on overall visual perception. 

## Task

Evaluate the consistency between the images according to the following scale (1 to 5):

- **5**: ONLY the changes explicitly required by the instruction are present. All other details are completely identical between the two images.

- **4**: Besides changes explicitly required by the instruction, the second image contains **1 significant** unintended change AND/OR **1-2 minor** unintended changes.

- **3**: Besides changes explicitly required by the instruction, the second image has **2-3 significant** unintended changes AND/OR **3-4 minor** unintended changes.

- **2**: Besides changes explicitly required by the instruction, the second image has **4+ significant** unintended changes AND/OR **5+ minor** unintended changes.

- **1**: The second image is almost entirely different from the original.

## Requirements

**CRITICAL - What Consistency Means**: 

- Consistency ONLY evaluates: "Did any changes occur that were NOT mentioned in the instruction?"
- It does NOT evaluate whether the instruction was successfully executed (that is evaluated separately).

**Exceptions - Do NOT count as inconsistencies**:

- **Occlusion effects**: Elements appearing/disappearing as a natural consequence of the instructed edit (e.g., background revealed when object is removed).
- **Image quality variations**: Small differences in sharpness, blur, noise, contrast, color temperature, lighting, reflection, shadow, saturation, etc. unless the instruction explicitly addresses these attributes.
- **Entity Replacement EXPLICITLY instructed by instruction**: When the instruction explicitly requires REPLACING entity A with B, ALL attributes of the new entity B are NOT consistency issues — only evaluate whether OTHER elements (background, other objects, scene composition) remain unchanged. NOTE: For ADD/REMOVE instructions, unintended entity removals/additions ARE inconsistencies. For Attribute Modification (e.g., change color, size, position), ONLY the specified attribute may change, any other changes in attributes of the same entity are inconsistencies.
- **Environmental changes**: Environmental changes that are a DIRECT PHYSICAL consequence of the instructed edit (e.g., lights turning on when changing daytime to night, wet ground when adding rain, shadows changing when lighting changes). Note: This does NOT include material substitutions/texture or object reposition/replacements that are merely aesthetically associated with the instruction.

Note: Apart from the exceptions listed above, other changes not explicitly instructed should be counted as inconsistencies."

## Output Format

You have to give your output in this way (Keep your reasoning concise and short.):
{{
"reasoning" : "<YOUR_REASONING>",
"score" : [1/2/3/4/5]
}}
"""


SCORE_EXECUTION = """
You are a highly skilled image evaluator. You will receive two images (an original image and a modified image) and a specific edit instruction. The second image is known to have been altered based on this instruction, starting from the first image. Your task is to evaluate the execution successfulness of the edit instruction.

Instruction: {prompt}

## Task

Evaluate the execution successfulness of the edited image according to the following scale (1 to 5):

- **5 (Perfect Execution)**: The edited image perfectly implements all aspects of the instruction. All requested changes are present and correctly executed.

- **4 (Good Execution)**: The edited image successfully implements all key aspects of the instruction, with only a very subtle missing detail that doesn't significantly affect whether the instruction was followed.

- **3 (Partial Execution)**: The edited image implements the main intent of the instruction, but some significant elements that was explicitly requested is missing or incorrectly implemented.

- **2 (Poor Execution)**: The edited image barely follows the instruction. Most requested changes are missing or incorrectly implemented, though there may be a vague attempt at following the instruction.

- **1 (No Execution)**: The edited image does not follow the instruction at all. No requested changes are visible, or the changes are completely contrary to what was requested.

**CRITICAL - Evaluation Scope**: 

- Only evaluate whether the REQUESTED changes are present and correctly implemented.
- Ignore any extra/unrequested modifications, rendering quality, realism, or unrelated consistency issues.

## Output Format

You have to give your output in this way (Keep your reasoning concise and short.):
{{
"reasoning" : "<YOUR_REASONING>",
"score" : [1/2/3/4/5]
}}
"""

