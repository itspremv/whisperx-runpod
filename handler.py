import runpod
import whisperx
import torch
import gc

# ============================================================================
# PyTorch 2.6+ Compatibility Fix
# ============================================================================
# PyTorch 2.6 changed the default of weights_only from False to True in torch.load()
# The pyannote.audio models use omegaconf classes which aren't in the default safe globals.
# This fix adds them to the allowlist so the models can be loaded safely.
# See: https://pytorch.org/docs/stable/generated/torch.load.html
# ============================================================================
try:
    from omegaconf import DictConfig, ListConfig, OmegaConf
    torch.serialization.add_safe_globals([DictConfig, ListConfig, OmegaConf])
    print("[handler] PyTorch 2.6+ compatibility fix applied for omegaconf")
except ImportError:
    print("[handler] omegaconf not installed - diarization may not work")
except Exception as e:
    print(f"[handler] Failed to apply PyTorch fix: {e}")


def handler(event):
    """
    WhisperX Handler with Speaker Diarization (pyannote 3.1)

    Input parameters:
        - audio_file (str, required): URL to the audio file
        - language (str, optional): Language code (e.g., 'en'). Auto-detected if not provided.
        - batch_size (int, optional): Batch size for transcription. Default: 16
        - align_output (bool, optional): Enable word-level alignment. Default: True
        - diarization (bool, optional): Enable speaker diarization. Default: False
        - huggingface_access_token (str, required if diarization=True): HuggingFace token for pyannote
        - min_speakers (int, optional): Minimum number of speakers for diarization
        - max_speakers (int, optional): Maximum number of speakers for diarization

    Output:
        - segments (list): List of transcription segments with timestamps and optional speaker labels
        - detected_language (str): Detected language code
        - language_probability (float): Confidence of language detection

    Fixed for PyTorch 2.6+ weights_only=True compatibility (2026-02-03)
    """
    try:
        input_data = event['input']

        # Extract parameters
        audio_file = input_data.get('audio_file')
        language = input_data.get('language')
        batch_size = input_data.get('batch_size', 16)

        if not audio_file:
            return {"error": "audio_file is required"}

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        print(f"[handler] Device: {device}, Compute type: {compute_type}")
        print(f"[handler] Audio file: {audio_file[:100]}...")
        print(f"[handler] Language: {language or 'auto-detect'}, Batch size: {batch_size}")

        # ====================================================================
        # Step 1: Load audio
        # ====================================================================
        print("[handler] Loading audio...")
        audio = whisperx.load_audio(audio_file)
        print(f"[handler] Audio loaded, duration: {len(audio)/16000:.1f}s")

        # ====================================================================
        # Step 2: Whisper Transcription
        # ====================================================================
        print("[handler] Loading Whisper model (large-v3-turbo)...")
        model = whisperx.load_model(
            "large-v3-turbo",
            device,
            compute_type=compute_type,
            language=language
        )

        print("[handler] Transcribing...")
        result = model.transcribe(
            audio,
            batch_size=batch_size,
            language=language
        )
        print(f"[handler] Transcription complete, detected language: {result.get('language')}")

        # Memory cleanup
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        # ====================================================================
        # Step 3: Alignment (for better word-level timestamps)
        # ====================================================================
        if input_data.get('align_output', True):
            print("[handler] Loading alignment model...")
            align_model, metadata = whisperx.load_align_model(
                language_code=result["language"],
                device=device
            )

            print("[handler] Aligning transcription...")
            result = whisperx.align(
                result["segments"],
                align_model,
                metadata,
                audio,
                device,
                return_char_alignments=False
            )
            print("[handler] Alignment complete")

            # Memory cleanup
            del align_model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        # ====================================================================
        # Step 4: Speaker Diarization (optional)
        # ====================================================================
        if input_data.get('diarization', False):
            hf_token = input_data.get('huggingface_access_token')

            if not hf_token:
                return {"error": "huggingface_access_token required for diarization"}

            print("[handler] Loading diarization model (pyannote)...")

            # Note: PyTorch 2.6+ fix is applied at module level above
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=hf_token,
                device=device
            )

            print("[handler] Running speaker diarization...")
            diarize_segments = diarize_model(
                audio,
                min_speakers=input_data.get('min_speakers'),
                max_speakers=input_data.get('max_speakers')
            )

            print("[handler] Assigning speakers to words...")
            result = whisperx.assign_word_speakers(diarize_segments, result)

            # Count unique speakers
            speakers = set()
            for seg in result.get("segments", []):
                if "speaker" in seg:
                    speakers.add(seg["speaker"])
            print(f"[handler] Diarization complete, found {len(speakers)} speakers")

            # Memory cleanup
            del diarize_model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        # ====================================================================
        # Step 5: Format response
        # ====================================================================
        segments = result.get("segments", [])
        print(f"[handler] Returning {len(segments)} segments")

        output = {
            "segments": segments,
            "detected_language": result.get("language"),
            "language_probability": result.get("language_probability")
        }

        return output

    except Exception as e:
        print(f"[handler] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# Start the RunPod serverless handler
runpod.serverless.start({"handler": handler})
