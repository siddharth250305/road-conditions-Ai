"""Small helpers shared by the export and API tests."""

import io

from PIL import Image

CLASSES = ["dry", "wet", "water", "fresh_snow", "melted_snow", "ice"]
TEMPERATURE = 1.7


def image_bytes(size=(360, 240), fmt="PNG", mode="RGB") -> bytes:
    """A small synthetic picture, encoded the way an upload would be."""
    image = Image.new(mode, size, (120, 110, 100) if mode == "RGB" else 128)
    for x in range(0, size[0], 8):  # some texture, so it is not a flat field
        for y in range(0, size[1], 8):
            image.putpixel(
                (x, y), (x * 7 % 256, y * 5 % 256, (x + y) % 256) if mode == "RGB" else 255
            )
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()
