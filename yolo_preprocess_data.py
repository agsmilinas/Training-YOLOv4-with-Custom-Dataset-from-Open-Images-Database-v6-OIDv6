
from os import chdir, path, listdir, getcwd
import shutil
import cv2
import sys

DIRS = ["train", "validation", "test"]
DEBUG = True

SKIP_TRANSLATE_LABELS = False
SKIP_GENERATE_FILE_LISTS = False
SKIP_GENERATE_OBJ_FILE = False

if len(sys.argv) < 2:
    print("Missing classes file as argument")
    raise SystemExit

classes_file = path.realpath(sys.argv[1])

def print_msg(msg, isDebug=False):
    if not isDebug:
        print(msg)
    elif isDebug and DEBUG:
        print(msg)

def get_classes(classes_file):
    with open(classes_file) as f:
        return [l.strip().lower().replace(" ", "_") for l in f.readlines()]

def label_contents(img_filename, classes):
    # It assumes is already in the img_filename directory and that label file
    # is on "labels" directory
    img = cv2.imread(img_filename)
    height, width, _ = img.shape
    file_class = "_".join(path.basename(img_filename).split("_")[:-1])
    class_idx = classes.index(file_class)
    # OIDv6 Label data
    label_file = img_filename[:-4] + ".txt"
    label_lines = [line.strip() for line in open("labels/" + label_file, "r")]
    new_lines = []
    for label_line in label_lines:
        label, x1, y1, x2, y2 = label_line.split()
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        box_width = x2 - x1
        box_height = y2 - y1
        center_x = x1 + (box_width / 2.0)
        center_y = y1 + (box_height / 2.0)
        relative_cx = center_x / width
        relative_cy = center_y / height
        relative_bw = box_width / width
        relative_bh = box_height / height
        new_lines.append('{0} {1} {2} {3} {4}'.format(
            class_idx, relative_cx, relative_cy, relative_bw, relative_bh))
    return "\n".join(new_lines)

chdir(path.join("OIDv6", "multidata"))

######## Translate Labels to YOLO format  ########

if not SKIP_TRANSLATE_LABELS:
    classes = get_classes(classes_file)

    for DIR in DIRS:
        chdir(DIR)
        image_files = []
        for filename in listdir():
            labels_file = path.join(getcwd(), filename[:-4] + ".txt")
            if (    path.isfile(filename)
                    and filename.endswith(".jpg")
                    and not path.isfile(labels_file) ):
                image_files.append(filename)
                labels_file_contents = label_contents(filename, classes)
                display_filename = DIR + "/" + path.basename(labels_file)
                print_msg("Generating Labels File " + display_filename)
                with open(labels_file, "w") as f:
                    f.write(labels_file_contents + "\n")
        class_list_file = path.join("..", DIR + ".txt")
        with open(class_list_file, "w") as f:
            for image in image_files:
                f.write(f"{DIR}/{image}\n")
        chdir("..")
    print_msg("\n\n================= Label Translation Finished =================\n\n")

if not SKIP_GENERATE_FILE_LISTS:
    for DIR in DIRS:
        chdir(DIR)
        file_list = path.join("..", DIR + ".txt")
        with open(file_list, "w") as f:
            for filename in listdir(getcwd()):
                if filename.endswith(".jpg"):
                    f.write(path.join(DIR, filename) + "\n")
        print_msg(f"File List {DIR}.txt generated")
        chdir("..")
    print_msg("\n\n================= File Lists Generation Finished =================\n\n")

if not SKIP_GENERATE_OBJ_FILE:
    chdir("..")
    classes_file = shutil.copy(classes_file, path.join(getcwd(), "classes.txt"))
    num_classes = sum(1 for line in open(classes_file))
    with open(path.join(getcwd(), "obj.data"), "w") as f:
        f.write(f"classes={num_classes}\n")
        f.write(f"train=multidata/train.txt\n")
        f.write(f"valid=multidata/validation.txt\n")
        f.write(f"names={classes_file}\n")
        f.write(f"backup=./\n")
    print_msg("\n\n================= Object File Generation Finished =================\n\n")
