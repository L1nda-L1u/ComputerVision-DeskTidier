from pathlib import Path

# Rename HEIC files
heic_dir = Path("DeskTidier Database")
heic_files = sorted(set(heic_dir.glob("*.heic")) | set(heic_dir.glob("*.HEIC")))
for i, f in enumerate(heic_files, start=1):
    new_name = heic_dir / f"desk_{i:03d}.heic"
    f.rename(new_name)
    print(f"{f.name} -> {new_name.name}")

# Rename JPG files
jpg_dir = Path("jpg_images")
jpg_files = sorted(set(jpg_dir.glob("*.jpg")) | set(jpg_dir.glob("*.JPG")))
for i, f in enumerate(jpg_files, start=1):
    new_name = jpg_dir / f"desk_{i:03d}.jpg"
    f.rename(new_name)
    print(f"{f.name} -> {new_name.name}")

print("Done.")
