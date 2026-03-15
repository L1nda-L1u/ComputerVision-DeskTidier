from pathlib import Path
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

input_dir = Path("DeskTidier Database")
output_dir = Path("jpg_images")
output_dir.mkdir(exist_ok=True)

for heic_file in list(input_dir.glob("*.heic")) + list(input_dir.glob("*.HEIC")):
    img = Image.open(heic_file)
    out_file = output_dir / (heic_file.stem + ".jpg")
    img.convert("RGB").save(out_file, "JPEG", quality=95)

print("Done.")
