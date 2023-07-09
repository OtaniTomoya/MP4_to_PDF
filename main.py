import argparse
import os
import cv2
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


class MP4_TO_PDF:
    def __init__(self, arg):
        self.output_path = os.path.splitext(arg)[0]
        self.video_path = self.output_path + '.mp4'
        self.threshold = 0.99
        self.pdf_path = self.output_path + '.pdf'

    def video_to_frames(self):
        os.makedirs(self.output_path, exist_ok=True)
        video = cv2.VideoCapture(self.video_path)
        success, frame = video.read()
        count = 0
        with tqdm(total=success) as pbar:
            while success:
                if count % 100 == 0:
                    cv2.imwrite(self.output_path + "/%d.jpg" % count, frame)
                pbar.update(1)
                success, frame = video.read()
                count += 1

        video.release()

    def extract_features(self, image_path):
        image = cv2.imread(image_path)
        feature_vector = image.flatten()
        return feature_vector

    def remove_duplicate_images(self):
        image_files = [file for file in os.listdir(self.output_path) if file.endswith((".jpg", ".jpeg", ".png"))]
        feature_vectors = []
        duplicate_indices = set()
        print('\n画像の特徴ベクトルを抽出')
        for image_file in tqdm(image_files):
            image_path = os.path.join(self.output_path, image_file)
            feature_vector = self.extract_features(image_path)
            feature_vectors.append(feature_vector)
        num_images = len(feature_vectors)

        print('\n重複した画像を削除')
        for i in tqdm(range(num_images - 1)):
            if i in duplicate_indices:
                continue
            for j in range(i + 1, num_images):
                if j in duplicate_indices:
                    continue
                similarity = cosine_similarity([feature_vectors[i]], [feature_vectors[j]])[0][0]
                if similarity >= self.threshold:
                    duplicate_indices.add(j)
        for index in duplicate_indices:
            image_file = image_files[index]
            image_path = os.path.join(self.output_path, image_file)
            os.remove(image_path)
        print(f'\n{len(duplicate_indices)}個の画像を削除しました')

    def images_to_pdf(self):
        image_paths = [file for file in os.listdir(self.output_path) if file.endswith((".jpg", ".jpeg", ".png"))]
        c = canvas.Canvas(self.pdf_path, pagesize=letter)
        idx = [int(image_path[:-4]) for image_path in image_paths]
        idx.sort()
        image_paths = [str(i) + '.jpg' for i in idx]
        for image_path in image_paths:
            image_path = f'{self.output_path}/{image_path}'
            c.drawImage(image_path, 0, 0, width=letter[0], height=letter[1])
            c.showPage()

        c.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="動画名前を記入")
    args = parser.parse_args()
    file_name = args.name
    print(f'{file_name}の変換を開始します')
    MtP = MP4_TO_PDF(file_name)
    MtP.video_to_frames()
    MtP.remove_duplicate_images()
    MtP.images_to_pdf()
