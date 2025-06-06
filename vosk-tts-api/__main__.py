import io
import sounddevice as sd
import soundfile as sf
from vosk_tts import Model, Synth

# Загружаем модель один раз при старте
print("Загрузка модели...")
model = Model(model_name="vosk-model-tts-ru-0.8-multi")
synth = Synth(model)
print("Модель загружена.")

# Функция синтеза
def speak(text, speaker_id=2):
    wav_bytes = synth.synth_bytes(text=text, speaker_id=speaker_id)
    return wav_bytes

# Пример использования
if __name__ == "__main__":
    speaker = 1  # сохраняем между итерациями
    while True:
        try:
            text = input(f"\n[Текущий спикер: {speaker}] Введите текст (или 'exit' / 'speaker'): ")

            if text.lower() == 'exit':
                break

            if text.lower() == 'speaker':
                text_speaker = input("Введите ID спикера (целое число): ")
                try:
                    speaker = int(text_speaker)
                    print(f"Спикер установлен на {speaker}")
                except ValueError:
                    print("Ошибка: ID спикера должен быть целым числом.")
                continue

            # Синтез речи
            wav_bytes = speak(text, speaker)
            buffer = io.BytesIO(wav_bytes)
            data, samplerate = sf.read(buffer, dtype='int16')

            # Воспроизводим через sounddevice
            sd.play(data, samplerate)
            sd.wait()

            print("Синтез завершён.")
        except Exception as e:
            print("Ошибка:", e)
