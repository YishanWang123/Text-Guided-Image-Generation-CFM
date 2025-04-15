# Text-Guided-Image-Generation-CFM

This project tackles text-guided image editing — generating an image that preserves the structure of the input while aligning with a given text prompt. The task combines elements of image-to-image translation and text-to-image synthesis.

I adopt a Conditional Flow Matching (CFM) model, where the text prompt is embedded via pretrained CLIP (openAIclip-vit-base-patch16). A pretrained VAE (stabilityai/sdxl-vae) is also used to compress images into a compact latent space, making generation more efficient and stable. To further improve controllability, we apply Classifier-Free Guidance (CFG), enabling stronger adherence to the text prompt without needing extra classifiers.

## Result

| Before and After                                                           |
|--------------------------------------------------------------------------|
| <img src="example/guided_dog.png" alt="Before and After" width="300" style="border:1px solid #ccc; border-radius:8px;" /> ||--------------------------------------------------------------------------|
| <img src="example/guided_cat.png" alt="Before and After" width="300" style="border:1px solid #ccc; border-radius:8px;" /> |

| Noise to Image                                                             |
|--------------------------------------------------------------------------|
| <img src="example/sample_11.gif" alt="Noise to Image" width="300" style="border:1px solid #ccc; border-radius:8px;" /> |

| Reconstructing Progress                                                     |
|----------------------------------------------------------------------------|
| <img src="example/guided_dog_progress.png" alt="Reconstructing Progress" width="300" style="border:1px solid #ccc; border-radius:8px;" /> |