Given a JSON input containing:
- An overall instruction for image generation
- Input and output image counts
- Detailed descriptions for each image (both input and output)
- A "panels" array specifying which images should be concatenated

Your task is to generate a cohesive multi-panel description that:

1. OVERALL STRUCTURE
- Begin with a comprehensive overview of all panels
- Use appropriate [TWO-PANEL], [THREE-PANEL], [FOUR-PANEL], etc. tag based on panel count
- Follow with individual [PANEL-X] descriptions
- Format as one continuous, flowing sentence using semicolons and commas
- Maintain logical flow between panels while preserving narrative continuity

2. RELATIONSHIP ANALYSIS
- Identify and track consistent elements across panels:
  * Character identities and names
  * Recurring objects or settings
  * Style consistency
  * Theme connections
  * Visual motifs
- Reference relationships using natural language:
  * "the same character"
  * "our protagonist"
  * "the familiar scene"
  * Do NOT use meta-references like "as seen in panel 1" or "similar to previous panel"

3. DETAIL PRESERVATION
- Maintain key visual elements from source descriptions:
  * Character appearances and clothing
  * Environmental details
  * Actions and poses
  * Color schemes and lighting
  * Emotional expressions
- Integrate these details naturally without breaking flow

4. STYLISTIC GUIDELINES
- Use rich, descriptive language that captures:
  * Visual composition
  * Emotional resonance
  * Movement and energy
  * Atmospheric qualities
- Maintain professional, clear writing style
- Avoid redundancy while preserving important details

5. TECHNICAL REQUIREMENTS
- Output must be valid JSON containing:
  * "explanation": Step-by-step analysis of your deduction process
  * "prompt": The generated multi-panel description

6. VALIDATION CHECKLIST
- Ensure all specified panels in the "panels" array are included
- Verify consistency in character references
- Check for natural flow between panel descriptions
- Confirm all critical details are preserved
- Validate proper tag usage and formatting
- Do not add details not present in the source descriptions
- Ensure the overview section is comprehensive enough to stand alone

Example Input Structure:
```json
{
    "instruction": "...",
    "input_image_count": "integer",
    "output_image_count": "integer",
    "descriptions": [
        {
            "image_id": "input_0001_<keywords>",
            "description": "..."
        },
        {
            "image_id": "input_0002_<keywords>",
            "description": "..."
        },
        ...
        {
            "image_id": "output_0001_<keywords>",
            "description": "..."
        },
        {
            "image_id": "output_0002_<keywords>",
            "description": "..."
        },
        ...
    ],
    "panels": {
        "panel_1": {
            "image_id": "input_0001_<keywords>",
            "description": "..."
        },
        "panel_2": {
            "image_id": "output_0002_<keywords>",
            "description": "..."
        },
        ...
    }
}
```

Example Output Structure:
```json
{
    "explanation": "1. Identified main character relationships...\n2. Analyzed visual consistency...\n3. Connected narrative elements...",
    "prompt": "In this [THREE-PANEL] image, [comprehensive overall description]; [PANEL-1] [first panel details], [PANEL-2] [second panel details], [PANEL-3] [third panel details]."
}
```

Prompt Example 1:
```text
In this [TWO-PANEL] image, a stylish winter portrait undergoes a transformation from a photograph to a digital illustration, showcasing subtle artistic interpretations and enhancements; [PANEL-1] the photograph features a person dressed warmly with a black beanie, sunglasses, a thick red and blue scarf, a white coat, and black leggings, holding a coffee cup and a paper bag while standing on a snowy street, exuding a cozy yet fashionable vibe, [PANEL-2] the illustration captures the same person with brighter, more vibrant colors, smoothened textures, and stylized shading, emphasizing clean lines and an artistic flair that highlights the accessories, such as the scarf and sunglasses, with a slightly exaggerated, cartoon-like appearance that retains the original's charm.
```

Prompt Example 2:
```text
This [THREE-PANEL] image captures the whimsical and curious nature of a young girl in various moments of contemplation and play; [PANEL-1] the first panel shows her intently focused on eating from a beautifully patterned bowl, her posture relaxed and captured in warm lighting that highlights her engagement with the meal, [PANEL-2] while the second panel reveals her resting her chin on her hands over a wooden fence, her eyes looking upwards dreamily, possibly lost in thought or daydreams as soft green foliage blurs the frame, [PANEL-3] and the final panel features her crouched amidst lush green plants, gazing slightly upwards with a playful and inquisitive expression, embodying the spirit of youthful adventure and exploration.
```

Remember:
- The goal is to create a description that can serve as a high-quality prompt for image generation while maintaining narrative coherence and visual consistency across all panels.
- STRICTLY adhere to the user-provided panel description, adding only consistent elements based on the context of instructions or other provided information.
- STRICTLY AVOID introducing any imagined or non-existent details that are not explicitly mentioned in the user’s description.