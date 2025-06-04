import os
import csv

def create_test_labels():
    test_dir = './dataset/test'
    label_file = './test_labels.csv'
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

    with open(label_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'label'])
        for fname in image_files:
            if fname.startswith('REAL'):
                label = 'real'
            elif fname.startswith('fake_image'):
                label = 'fake'
            else:
                continue  # skip files that don't match either pattern
            writer.writerow([fname, label])

if __name__ == "__main__":
    create_test_labels() 