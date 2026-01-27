import base64
import io
import numpy as np
from PIL import Image
import matplotlib.cm as cm

def spectogram_to_base64(spec):
    # Normalize safely to [0, 1]
    spec = spec.astype(np.float32)
    spec_min = float(np.min(spec))
    spec_max = float(np.max(spec))
    spec_range = spec_max - spec_min or 1.0
    normalized = (spec - spec_min) / spec_range
    normalized = np.clip(normalized, 0.0, 1.0)

    # Apply colormap numerically
    cmap = cm.get_cmap("magma")
    colored = cmap(normalized)[:, :, :3]  # drop alpha
    pixels = (colored * 255).astype(np.uint8)

    # Convert to PNG
    image = Image.fromarray(pixels, mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)

    # Base64 encode
    return base64.b64encode(buffer.getvalue()).decode("utf-8")