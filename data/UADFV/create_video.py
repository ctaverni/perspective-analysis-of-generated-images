import os
import cv2

def frames_to_video(frames_folder, output_folder='videos', fps=30):
    if not os.path.isdir(frames_folder):
        raise ValueError(f"Le chemin spécifié n'est pas un dossier : {frames_folder}")
    
    os.makedirs(output_folder, exist_ok=True)
    
    for subfolder in sorted(os.listdir(frames_folder)):
        subfolder_path = os.path.join(frames_folder, subfolder)
        
        if not os.path.isdir(subfolder_path):
            continue  # Ignore les fichiers non dossiers
        
        # Liste et trie les fichiers image
        images = sorted([
            f for f in os.listdir(subfolder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        if not images:
            print(f"Aucune image trouvée dans {subfolder_path}, saut...")
            continue

        # Lit la première image pour récupérer les dimensions
        first_image_path = os.path.join(subfolder_path, images[0])
        frame = cv2.imread(first_image_path)
        if frame is None:
            print(f"Impossible de lire la première image dans {subfolder_path}")
            continue
        height, width, layers = frame.shape

        # Nettoie le nom du dossier pour générer le nom de la vidéo
        video_name = subfolder
        if video_name.endswith('_fake'):
            video_name = video_name[:-5]  # Retire les 5 derniers caractères (_fake)

        video_path = os.path.join(output_folder, f"{video_name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        for image_file in images:
            image_path = os.path.join(subfolder_path, image_file)
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Image invalide ignorée : {image_path}")
                continue
            video.write(frame)

        video.release()
        print(f"Vidéo créée : {video_path}")

if __name__ == "__main__":
    frames_to_video("fake/frames", fps=30, output_folder='videos/fake')
    frames_to_video("real/frames", fps=30, output_folder='videos/real')

