import numpy as np
import sounddevice as sd
import onnxruntime as ort
from scipy.signal import resample_poly
from scipy.io import wavfile
from queue import Queue
from collections import deque
from time import sleep
import subprocess
from threading import Thread
from os.path import join
from datetime import datetime
import time

MODEL_PATH = 'models/silero-vad/silero_vad_half.onnx'
AUDIO_FOLDER = 'recorded_audio'

class SileroVad:
    def __init__(self):

        self.session = ort.InferenceSession(MODEL_PATH)
        self.threshold = 0.5
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
    
    def __call__(self, audio_chunk: np.ndarray) -> bool:
        '''Retorna True si hay habla y false si no'''

        audio_chunk = np.array(audio_chunk, dtype=np.float32)
        audio_chunk = audio_chunk.reshape(1, -1)

        inputs = {
            "input": audio_chunk,
            "state": self.state,
            #"sr": np.array(16000, dtype=np.int64)
        }
        
        outputs = self.session.run(None, inputs)

        speech_prob = outputs[0][0][0]

        self.state = outputs[1]

        if speech_prob >= self.threshold:
            return True
        else:
            return False
    

class MP3Writer:
    def __init__(self):
        self.sample_rate = 16000

    def __call__(self, audio: np.ndarray)-> bool:

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S%f")
        filename = f"{timestamp}.mp3"
        file_path = join(AUDIO_FOLDER, filename)

        process = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-f", "f32le",
                "-ar", str(self.sample_rate),
                "-ac", "1",
                "-i", "pipe:0",
                file_path
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        process.stdin.write(audio.astype(np.float32).tobytes())
        process.stdin.close()
        process.wait()
        print(f'audio guardado: {filename}')

class AudioQueue:
    def __init__(self, 
                 before_duration_ms=2000,
                 after_duration_ms=500, 
                 chunk_size=1588, 
                 samplerate= 16000):
        
        self.num_chunks_before = int((before_duration_ms / 1000) * samplerate / chunk_size)
        self.num_chunks_after = int((after_duration_ms / 1000) * samplerate / chunk_size)
    
        self._count_chunks_silence = 0 #contador de chunks de silencio para determinar cuándo cortar
        self._count_chunks_speech = 0 #contador de chunks de habla para determinar cuándo empezar a grabar
        self._threshold_chunks_true = 10 #ajustable (controla cuántos chunks de habla seguidos se necesitan para empezar a grabar)
        self._threshold_chunks_false = 15 #ajustable (controla cuántos chunks de silencio seguidos se necesitan para cortar)
        self.recording = False

        self.buffer = Queue() # buffer principal donde soundevice pone los chunks de audio
        self._queue_speech_before = deque(maxlen=self.num_chunks_before+ self._threshold_chunks_true)
        self._queue_record = Queue()

        self.silero_vad = SileroVad()
        self.writer = MP3Writer()
    
    def __call__(self, chunk: np.ndarray):
        self.buffer.put(chunk)

    def loop(self):
        while True:
            chunk = self.buffer.get()
            chunk = self._chunk_prepocess(chunk)

            is_speech = self.silero_vad(chunk[:576])

            if is_speech:
                self._count_chunks_silence = 0
                if self.recording:
                    self._queue_record.put(chunk)
                elif self._count_chunks_speech >= self._threshold_chunks_true:
                    print('Audio detectado, empezando a grabar...')
                    self.recording = True
                    self._queue_record.put(chunk)
                    self._count_chunks_speech = 0
                else:
                    self._count_chunks_speech += 1
                    self._queue_speech_before.append(chunk)
            else:
                self._count_chunks_speech = 0
                if not self.recording:
                    self._queue_speech_before.append(chunk)
                elif self._count_chunks_silence >= self._threshold_chunks_false:
                    self.recording = False
                    self._count_chunks_silence = 0
                    self._save_audio()
                    
                else:
                    self._count_chunks_silence += 1
                    self._queue_record.put(chunk)
    
    def _chunk_prepocess(self, chunk: np.ndarray) -> np.ndarray:
        chunk = chunk.flatten()
        chunk = resample_poly(chunk, 160, 441)
        return chunk
        
    def _save_audio(self):
        recorded_audio = list(self._queue_speech_before) + list(self._queue_record.queue)
        self.writer(np.array(recorded_audio))
        self._queue_record = Queue() # resetear el buffer de grabación
        self._queue_speech_before.clear() # limpiar el buffer de pre-speech
    
    def start(self):
        thread = Thread(target=self.loop, daemon=True)
        thread.start()


class AudioRecorder:

    def __init__(self):

        self.audio_queue = AudioQueue()
        self.audio_queue.start()
    
    def callback(self, indata, frames, time, status):
        if status:
            print(status)

        self.audio_queue(indata.copy())

    def start(self):
        with sd.InputStream(channels=1, 
                            samplerate=44100, 
                            blocksize=2205, 
                            callback=self.callback,
                            device= 'USB Audio Device'):

            print("Grabando... Presiona Ctrl+C para detener.")
            while True:
                sleep(1)
    
if __name__ == "__main__":
    AudioRecorder().start()