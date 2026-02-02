import runpod
import whisperx
import torch
import gc

def handler(event):
    """
    WhisperX Handler mit Speaker Diarization (pyannote 3.1)
    """
    try:
        input_data = event['input']
        
        # Parameter
        audio_file = input_data.get('audio_file')
        language = input_data.get('language')
        batch_size = input_data.get('batch_size', 16)
        
        if not audio_file:
            return {"error": "audio_file is required"}
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        # 1. Audio laden
        audio = whisperx.load_audio(audio_file)
        
        # 2. Whisper Transcription
        model = whisperx.load_model(
            "large-v3-turbo", 
            device, 
            compute_type=compute_type,
            language=language
        )
        
        result = model.transcribe(
            audio, 
            batch_size=batch_size,
            language=language
        )
        
        # Memory cleanup
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        # 3. Alignment (für bessere Timestamps)
        if input_data.get('align_output', True):
            align_model, metadata = whisperx.load_align_model(
                language_code=result["language"], 
                device=device
            )
            
            result = whisperx.align(
                result["segments"], 
                align_model, 
                metadata, 
                audio, 
                device,
                return_char_alignments=False
            )
            
            # Memory cleanup
            del align_model
            gc.collect()
            torch.cuda.empty_cache()
        
        # 4. Speaker Diarization (optional)
        if input_data.get('diarization', False):
            hf_token = input_data.get('huggingface_access_token')
            
            if not hf_token:
                return {"error": "huggingface_access_token required for diarization"}
            
            # Diarization Pipeline (pyannote 3.1)
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=hf_token,
                device=device
            )
            
            diarize_segments = diarize_model(
                audio,
                min_speakers=input_data.get('min_speakers'),
                max_speakers=input_data.get('max_speakers')
            )
            
            result = whisperx.assign_word_speakers(diarize_segments, result)
            
            # Memory cleanup
            del diarize_model
            gc.collect()
            torch.cuda.empty_cache()
        
        # 5. Response aufbereiten
        output = {
            "segments": result.get("segments", []),
            "detected_language": result.get("language"),
            "language_probability": result.get("language_probability")
        }
        
        return output
        
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
