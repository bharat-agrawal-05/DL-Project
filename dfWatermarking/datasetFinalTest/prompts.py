# generate_sample_prompts.py
import random

def generate_prompts_file(num_prompts: int, output_filename: str = "prompts.txt"):
    """Generates a sample prompts file."""
    subjects = ["majestic mountain", "serene lake", "ancient forest", "bustling cityscape", "futuristic vehicle",
                "mythical creature", "abstract sculpture", "cozy cottage", "vast desert", "tropical island",
                "steampunk airship", "glowing mushroom", "enchanted garden", "hidden waterfall", "starry night sky",
                "cyberpunk alley", "medieval castle", "art deco building", "robot companion", "steaming jungle"]
    styles = ["photorealistic", "oil painting", "watercolor", "impressionistic", "fantasy art", "sci-fi concept art",
              "pixel art", "cel-shaded", "detailed illustration", "minimalist", "surreal", "gothic", "vaporwave",
              "low poly", "chalk drawing"]
    moods = ["dreamy", "epic", "tranquil", "mysterious", "vibrant", "ominous", "joyful", "melancholic",
             "dynamic", "peaceful", "eerie", "nostalgic", "ethereal", "dramatic"]
    lighting = ["golden hour", "moonlit", "neon lights", "soft daylight", "dramatic shadows", "bioluminescent glow",
                "sunrise", "misty morning", "underwater lighting", "volumetric lighting"]

    prompts = set()
    while len(prompts) < num_prompts:
        subject = random.choice(subjects)
        style = random.choice(styles)
        mood = random.choice(moods)
        light = random.choice(lighting)
        # Vary prompt structure for more diversity
        if random.random() < 0.3:
            prompt = f"{style} of a {mood} {subject}, {light}"
        elif random.random() < 0.6:
            prompt = f"A {mood} {subject} in {style} style, {light}"
        else:
            prompt = f"{subject}, {style}, {mood}, {light}"
        prompts.add(prompt)

    with open(output_filename, "w") as f:
        for p in list(prompts)[:num_prompts]: # Ensure exactly num_prompts if set size is larger
            f.write(p + "\n")
    print(f"Generated {min(len(prompts), num_prompts)} prompts in {output_filename}")

if __name__ == "__main__":
    generate_prompts_file(1000, "TokenPatternsDataset/prompts.txt") # Save inside the dataset directory