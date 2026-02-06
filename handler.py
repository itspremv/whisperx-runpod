"""
WhisperX Handler with Speaker Diarization (Pyannote 3.1)

REQUIREMENTS BEFORE USING DIARIZATION:
1. Accept HuggingFace model licenses:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
2. Pass your HuggingFace Access Token in the request

FIXED (2026-02-06):
- WhisperX 3.3.4+ API change: DiarizationPipeline moved to whisperx.diarize module
- See: https://github.com/m-bain/whisperX/issues/1131
"""

import runpod
import whisperx
import torch
import gc
import traceback

# Import DiarizationPipeline from the correct location (WhisperX 3.3.4+)
# Old: whisperx.DiarizationPipeline (removed in 3.3.4)
# New: whisperx.diarize.DiarizationPipeline
try:
    from whisperx.diarize import DiarizationPipeline
    print("[handler] DiarizationPipeline imported from whisperx.diarize (WhisperX 3.3.4+)")
except ImportError:
    # Fallback for older versions
    try:
        DiarizationPipeline = whisperx.DiarizationPipeline
        print("[handler] DiarizationPipeline imported from whisperx (legacy)")
    except AttributeError:
        DiarizationPipeline = None
        print("[handler] WARNING: DiarizationPipeline not available - diarization will fail")

# ============================================================================
# PyTorch 2.6+ Compatibility Fix
# ============================================================================
try:
    from omegaconf import DictConfig, ListConfig, OmegaConf
    # Only apply if PyTorch >= 2.6
    if hasattr(torch.serialization, 'add_safe_globals'):
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
    """
    try:
        input_data = event.get('input', {})

        # Extract parameters
        audio_file = input_data.get('audio_file')
        language = input_data.get('language')
        batch_size = input_data.get('batch_size', 16)
        enable_diarization = input_data.get('diarization', False)

        if not audio_file:
            return {"error": "audio_file is required"}

        # Validate HuggingFace token early if diarization is requested
        hf_token = input_data.get('huggingface_access_token')
        if enable_diarization and not hf_token:
            return {
                "error": "huggingface_access_token is required for diarization. "
                         "Get your token from https://huggingface.co/settings/tokens and ensure you have "
                         "accepted the model licenses at: "
                         "https://huggingface.co/pyannote/speaker-diarization-3.1 and "
                         "https://huggingface.co/pyannote/segmentation-3.0"
            }

        # Check if DiarizationPipeline is available
        if enable_diarization and DiarizationPipeline is None:
            return {
                "error": "DiarizationPipeline not available in this WhisperX version. "
                         "Please update the RunPod handler to use whisperx.diarize.DiarizationPipeline"
            }

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        print(f"[handler] Device: {device}, Compute type: {compute_type}")
        print(f"[handler] PyTorch version: {torch.__version__}")
        print(f"[handler] WhisperX version: {whisperx.__version__ if hasattr(whisperx, '__version__') else 'unknown'}")
        print(f"[handler] Audio file: {audio_file[:100]}...")
        print(f"[handler] Language: {language or 'auto-detect'}, Batch size: {batch_size}")
        print(f"[handler] Diarization: {enable_diarization}, HF Token present: {bool(hf_token)}")

        # ====================================================================
        # Step 1: Load audio
        # ====================================================================
        print("[handler] Loading audio...")
        try:
            audio = whisperx.load_audio(audio_file)
        except Exception as e:
            return {"error": f"Failed to load audio: {str(e)}. Ensure the URL is publicly accessible."}

        print(f"[handler] Audio loaded, duration: {len(audio)/16000:.1f}s")

        # ====================================================================
        # Step 2: Whisper Transcription
        # ====================================================================
        print("[handler] Loading Whisper model (large-v3-turbo)...")
        try:
            model = whisperx.load_model(
                "large-v3-turbo",
                device,
                compute_type=compute_type,
                language=language
            )
        except Exception as e:
            return {"error": f"Failed to load Whisper model: {str(e)}"}

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
            try:
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
            except Exception as e:
                print(f"[handler] WARNING: Alignment failed: {str(e)}. Continuing without alignment.")

        # ====================================================================
        # Step 4: Speaker Diarization (optional)
        # ====================================================================
        if enable_diarization:
            print("[handler] Loading diarization model (pyannote)...")
            print("[handler] This requires accepted licenses at huggingface.co/pyannote/...")

            try:
                # Use the imported DiarizationPipeline (from whisperx.diarize in 3.3.4+)
                diarize_model = DiarizationPipeline(
                    use_auth_token=hf_token,
                    device=device
                )
            except Exception as e:
                error_msg = str(e)
                if "401" in error_msg or "authentication" in error_msg.lower():
                    return {
                        "error": f"HuggingFace authentication failed. Please ensure: "
                                 f"1) Your token is valid (https://huggingface.co/settings/tokens), "
                                 f"2) You have accepted the licenses at "
                                 f"https://huggingface.co/pyannote/speaker-diarization-3.1 and "
                                 f"https://huggingface.co/pyannote/segmentation-3.0. "
                                 f"Original error: {error_msg}"
                    }
                elif "403" in error_msg or "access" in error_msg.lower():
                    return {
                        "error": f"Access denied to Pyannote models. Please accept the model licenses at: "
                                 f"https://huggingface.co/pyannote/speaker-diarization-3.1 and "
                                 f"https://huggingface.co/pyannote/segmentation-3.0. "
                                 f"Original error: {error_msg}"
                    }
                else:
                    return {"error": f"Failed to load diarization model: {error_msg}"}

            print("[handler] Running speaker diarization...")
            try:
                diarize_segments = diarize_model(
                    audio,
                    min_speakers=input_data.get('min_speakers'),
                    max_speakers=input_data.get('max_speakers')
                )
            except Exception as e:
                return {"error": f"Speaker diarization failed: {str(e)}"}

            print("[handler] Assigning speakers to words...")
            result = whisperx.assign_word_speakers(diarize_segments, result)

            # Count unique speakers
            speakers = set()
            for seg in result.get("segments", []):
                if "speaker" in seg:
                    speakers.add(seg["speaker"])
            print(f"[handler] Diarization complete, found {len(speakers)} speakers: {speakers}")

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

        # Log a sample of the output for debugging
        if segments:
            sample = segments[0]
            print(f"[handler] Sample segment: speaker={sample.get('speaker', 'N/A')}, "
                  f"text={sample.get('text', '')[:50]}...")

        output = {
            "segments": segments,
            "detected_language": result.get("language"),
            "language_probability": result.get("language_probability")
        }

        return output

    except Exception as e:
        print(f"[handler] ERROR: {str(e)}")
        traceback.print_exc()
        return {"error": str(e)}


# Start the RunPod serverless handler
runpod.serverless.start({"handler": handler})
