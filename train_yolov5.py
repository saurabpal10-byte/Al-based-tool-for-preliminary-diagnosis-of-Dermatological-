import argparse, os, yaml, shutil
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# This script builds a detection dataset from classification folders by assigning one pseudo box per image.
# Replace with real annotations if available.

def make_yolo_dataset(cls_root: Path, out_root: Path):
    # YOLO format: images/ and labels/ with .txt per image: class_id x_center y_center w h (normalized)
    images_dir = out_root/"images"
    labels_dir = out_root/"labels"
    for split in ["train","val","test"]:
        (images_dir/split).mkdir(parents=True, exist_ok=True)
        (labels_dir/split).mkdir(parents=True, exist_ok=True)

    class_names = sorted([d.name for d in (cls_root/"train").iterdir() if d.is_dir()])
    class_to_id = {c:i for i,c in enumerate(class_names)}

    for split in ["train","val","test"]:
        for c in class_names:
            for img_path in (cls_root/split/c).rglob("*"):
                if img_path.suffix.lower() not in [".png",".jpg",".jpeg",".bmp",".tif",".tiff"]:
                    continue
                dst_img = images_dir/split/img_path.name
                shutil.copy2(img_path, dst_img)

                try:
                    with Image.open(img_path) as im:
                        w,h = im.size
                except:
                    w=h=1000
                # pseudo box = centered 80% of image
                xc, yc = 0.5, 0.5
                bw, bh = 0.8, 0.8
                label_line = f"{class_to_id[c]} {xc} {yc} {bw} {bh}\n"
                (labels_dir/split/dst_img.with_suffix(".txt").name).write_text(label_line)

    return class_names

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data_splits (classification)")
    ap.add_argument("--out", default="runs/yolo")
    ap.add_argument("--epochs", type=int, default=50)
    args = ap.parse_args()

    data_cls = Path(args.data)
    yolo_ds = Path(args.out)/"yolo_ds"
    yolo_ds.mkdir(parents=True, exist_ok=True)
    names = make_yolo_dataset(data_cls, yolo_ds)

    data_yaml = {
        "path": str(yolo_ds.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names
    }
    yaml_path = Path(args.out)/"data.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data_yaml, f)

    model = YOLO("yolov8n.pt")  # lightweight; behaves like YOLOv5 for our purposes
    model.train(data=str(yaml_path), epochs=args.epochs, imgsz=640, project=args.out, name="exp")
    print("Training complete.")

if __name__ == "__main__":
    main()