MODEL_PROMPT = """You are an expert Image Evaluator. 
Your task is to evaluate a generated image strictly based on the Original Prompt.

### Tasks
1. Before writing, carefully inspect the image in full. Do not rush.
2. Identify all explicit and implicit requirements from the Original Prompt.
   This includes, but is not limited to, elements such as main subjects, attributes, actions,
   relationships, style, composition, and any negative constraints.
3. Perform a step-by-step evaluation by assessing whether the image satisfies each identified requirement.
4. Assign a final alignment rating according to the rating scale below.

### Rating Scale
- **5**: All requirements, details, styles, and negative constraints are correct.
- **4**: Main content is correct, but 1-2 non-critical details and requirements are slightly off.
- **3**: Main subject(s) is present, but multiple requirements and details are missing.
- **2**: The majority of main subject(s) are missing or incorrect, though a small portion of the content remains relevant.
- **1**: Image is irrelevant to the original prompt.

### Output Format
Produce the output in **plain text**, strictly following the structure below:

Begin with:
Let's evaluate the image against the Original Prompt:

1. **Identified Requirement 1**:
- [Analysis...]

2. **Identified Requirement 2**:
- [Analysis...]

(Continue until all major requirements inferred from the prompt are evaluated)

**Final Analysis**:
[A concise summary paragraph explaining the final decision and why the specific rating was chosen.]

**Final Alignment Rating: [Rating]**
\\boxed{{[Rating]}}

### Constraints
1. The [Rating] inside \\boxed{{}} must be one of: 5, 4, 3, 2, 1.
2. Maintain objectivity. Treat all identified requirements as a strict checklist and evaluate each one accordingly.
"""



Quality_PROMPT = """You are a highly skilled image evaluator.

You will receive ONE generated image. 
Your task is to evaluate the **visual quality of this image ONLY**.

## Task
Evaluate the image across three dimensions on a 1–5 scale each:

**Important constraints:**
- Do NOT consider instruction-following or any external description (there is none).
- Base scores solely on what is visible in the single image.
- For stylized, non-photorealistic, or fantastical scenes, judge criteria by internal coherence within the depicted style.

### 1. Aesthetics (score1):
Focuses on composition design, visual elements, emotional impact, and overall visual appeal. It answers whether the image is "beautiful" and artistically balanced.
- **5**: Masterful composition, striking visual balance, harmonious color palettes, and strong artistic/emotional appeal.
- **4**: Highly pleasing and well-composed, good use of visual elements, but lacks a profound "wow" factor.
- **3**: Average composition; acceptable but somewhat plain, unmemorable, or slightly unbalanced.
- **2**: Poor composition, visually confusing, cluttered, or jarring color schemes.
- **1**: Visually chaotic, extremely unbalanced, or distinctly aesthetically displeasing.

### 2. Quality (score2):
Focuses on perceptual fidelity and the absence of degradation factors. It answers whether the image is "technically up to standard" (e.g., no noise, blur, compression).
- **5**: Pristine technical quality. Sharp, clear, with no visible noise, blur, ringing, or compression artifacts.
- **4**: Minor technical flaws (mild noise, slight blur, slight banding) that are only visible upon close inspection.
- **3**: Moderate artifacts (noticeable compression blocks, ghosting, seam lines, or inpainting boundaries) that affect overall clarity.
- **2**: Noticeable degradations (heavy distortion, severe noise, color bleeding, over/under-sharpening).
- **1**: Severe degradations (extreme distortion, prominent watermarks, heavily corrupted regions, unreadable technical quality).

### 3. Structure & Texture (score3):
Focuses on local features, geometric regularity, material properties (e.g., smoothness, roughness), detail richness, and scene complexity.
- **5**: Intricate and rich details, flawless geometric regularity, highly coherent material properties, perfect physics/anatomy/lighting within its style.
- **4**: Solid structure and rich textures, but with minor inconsistencies (e.g., slight warped geometry, minor lighting/material mismatch).
- **3**: Noticeable structural or textural issues (incorrect shadows/reflections, noticeable anatomical flaws, missing detail, flat/inconsistent textures).
- **2**: Severe structural failures (broken anatomy/limbs, impossible overlaps, severely mismatched depth/textures).
- **1**: Completely incoherent geometry, physics, or material properties; unrecognizable structures.

**Evaluation procedure:**
1. Assess **Aesthetics**: composition, color harmony, visual balance, emotional expression.
2. Identify **Quality** degradations: noise, blur, blockiness, banding, moiré, halos, edge fringing, chromatic aberration.
3. Inspect **Structure & Texture**: lighting/shadows, perspective/depth, occlusions, anatomy/poses, reflections, material/texture consistency, local detail richness and overall scene complexity.

**Output format:**
Provide the scores in a list: `[score1, score2, score3]`, where score1 is Aesthetics, score2 is Quality, and score3 is Structure & Texture. Keep your reasoning concise and short.

Output STRICTLY in the following JSON format:
{
  "score" : [score1, score2, score3],
  "reasoning" : "..."
}
"""


SCORE = (
    MODEL_PROMPT
    + "\n\n### Original Prompt\n{prompt}\n"
    + "### Additional Requirement (if any)\n{requirement}\n"
)
