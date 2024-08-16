from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import os
from model_train.face_detection.retrain import extract_face_embeddings, update_embedding_database, retrain_model
from model_train.face_detection.embedding_extract import embedder

class NewDriverHandler(FileSystemEventHandler):
    def __init__(self, embedder, database_file='embeddings.json'):
        self.embedder = embedder
        self.database_file = database_file

    def process(self, event):
        if event.event_type == 'created' and not event.src_path.endswith('.json'):
            driver_id = os.path.basename(os.path.dirname(event.src_path))
            print(f"New image detected: {event.src_path}")
            new_image_paths = [event.src_path]
            self.update_embeddings(new_image_paths, driver_id)

    def update_embeddings(self, image_paths, driver_id):
        new_embeddings = extract_face_embeddings(image_paths, self.embedder)
        update_embedding_database(new_embeddings, driver_id, self.database_file)
        retrain_model(self.database_file)
        print(f"Updated embeddings and retrained model for driver ID: {driver_id}")

    def on_created(self, event):
        self.process(event)

def monitor_directory(directory_to_watch, embedder, database_file='embeddings.json'):
    event_handler = NewDriverHandler(embedder, database_file)
    observer = Observer()
    observer.schedule(event_handler, path=directory_to_watch, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# Example usage
if __name__ == "__main__":
    directory_to_watch = 'data/new_drivers'
    monitor_directory(directory_to_watch, embedder)
